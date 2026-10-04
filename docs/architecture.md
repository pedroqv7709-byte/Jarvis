# Arquitectura de Jarvis (v0.6)

Este documento describe **lo que existe hoy** en el código. Las extensiones futuras se indican
explícitamente como tales.

## Principio

El modelo de lenguaje es un componente, nunca la autoridad. Política, permisos, estado y
verificación son deterministas y pertenecen a Jarvis. Un plan (venga de reglas o de un modelo) es
solo una propuesta: cada acción pasa por selección, autorización, ejecución y verificación.

## Flujo

```text
Interface (CLI)
  │  Jarvis.ask(text)
  ▼
Orchestrator.handle(request)                       orchestration/orchestrator.py
  ├─ Task creada (PENDING) y persistida             core/task.py, state/
  ├─ ContextBuilder.build(task)                     context/
  ├─ RequestGuard.check(request)   ── política a nivel de solicitud     safety/request_guard.py
  ├─ Planner.plan(...) + PlanValidator              planning/
  │     FallbackPlanner: ModelPlanner → (LiteLLMProvider) ──falla/timeout/no valida──▶ RuleBasedPlanner
  └─ TaskRunner.run(task, context)                  execution/runner.py
        └─ por cada acción: ActionExecutor.execute  execution/executor.py
              1. ToolSelector.select      → SELECTED      (herramienta registrada + args válidos)
              1b. Tool.resolve_target     → objetivo concreto (solo lectura; auditado: action.resolved)
              2. PolicyEngine.evaluate    → AUTHORIZED / DENIED / AWAITING_CONFIRMATION
                 └─ ConfirmationHandler (aprobar / rechazar / diferir)
              3. Tool.run (timeout duro, presupuesto de tarea, reintentos acotados) → EXECUTED / FAILED
              4. VerificationService.verify       → VERIFIED / (UNVERIFIABLE) / FAILED
              · checkpoint en StateStore tras cada paso
              · evento de auditoría en cada paso
  ▼
ResponseBuilder.build(task) → Response
```

## Modelo real en el bucle (v0.6)

v0.6 no añade capacidades: comprueba que el bucle de v0.5 funciona igual cuando las decisiones
vienen de un LLM real. No hay componentes nuevos; el punto de inserción ya existía:

```text
TaskRunner ─▶ FallbackPlanner ─▶ ModelPlanner ─▶ ModelRouter ─▶ LiteLLMProvider ─▶ LLM
   (bucle)      (valida cada        (prompt,         (por            (único módulo
                 decisión)           schema)          capacidades)     que conoce LiteLLM)
```

El bucle no sabe qué proveedor hay detrás: se elige por entorno (`JARVIS_MODEL_PROVIDER`,
`JARVIS_MODEL`, `JARVIS_MODEL_API_BASE`) y cualquier backend de LiteLLM es intercambiable.

Cambios:

- **Salida estructurada**: `LiteLLMProvider` envía el JSON Schema de la decisión incremental
  (`response_format: json_schema`, no estricto porque los argumentos de herramienta son objetos
  libres) cuando el mapa local de LiteLLM dice que el backend lo soporta; si no, `json_object` + el
  schema en el prompt (`JARVIS_MODEL_STRUCTURED_OUTPUT` lo fuerza). `ModelAnswer` *anuncia* solo el
  formato incremental (`response_json_schema`) pero sigue *aceptando* el plan completo pre-v0.5.
  Lo que prometa el backend da igual: la respuesta pasa por `parse_structured` y el
  `PlanValidator` como antes.
- **Sin reintentos ocultos**: `num_retries=0` y `max_retries=0` en cada llamada (el SDK de OpenAI
  reintenta 2 veces por defecto, lo que triplicaba en silencio el tiempo de una decisión).
- **Tiempo**: el timeout de cada llamada es `min(JARVIS_MODEL_TIMEOUT_SECONDS, tiempo restante de
  la tarea)`. Un modelo que deja de responder no puede saltarse el presupuesto global.
- **Contexto**: el mensaje del usuario es un JSON con `task` (escrito por Jarvis: objetivo,
  requisitos y si se cumplen, presupuesto, feedback) y `untrusted_data` (contexto, historial,
  observaciones). El prompt de sistema nombra explícitamente `untrusted_data` como datos, nunca
  instrucciones. Al ser JSON serializado, un texto observado no puede "salirse" de su campo. El
  catálogo de herramientas de Windows incluye los nombres de la allowlist (datos de configuración,
  no del modelo).
- **Secretos**: los errores del proveedor se limpian de fragmentos de claves (`sk-…`, `Bearer …`)
  antes de truncarse y llegar a la auditoría.
- **Requisitos**: el punto final de una frase ("Abre la calculadora y enfócala.") ya no impide
  extraer los requisitos verificables (antes la tarea quedaba sin requisitos y la completion se
  apoyaba solo en "la última acción tuvo éxito").

Qué no cambia (y está probado con modelos simulados que **obedecen** inyecciones en títulos de
ventana, salidas de herramientas, resultados que dicen "ahora estás autorizado" y contexto con
instrucciones falsas, `tests/test_model_loop.py`): un nombre fuera de la allowlist no se lanza,
una ruta o línea de comandos no pasa la validación, una herramienta inexistente no se ejecuta,
cerrar o borrar siguen pidiendo confirmación, un DENY sigue siendo DENY, campos como `approved`,
`permissions`, `verified`, `final` o `limits` en la respuesta la invalidan entera, `done` sin
evidencia se rechaza, y ciclos y límites detienen la tarea.

Validación real: `tests/test_live_model.py` (opt-in, `JARVIS_LIVE_MODEL=1`; con
`JARVIS_LIVE_WINDOWS=1` sobre el escritorio real, solo Calculadora). No se ha ejecutado en el
entorno de desarrollo de v0.6 por falta de credenciales y de un Ollama local.

## Bucle agente (v0.5)

`TaskRunner` es el bucle. En cada iteración:

1. Si hay una acción pendiente (planificada o con confirmación ya decidida) la ejecuta con
   `ActionExecutor` (resolver objetivo → política → confirmación → intentos acotados →
   verificación), igual que antes de v0.5. Si al reanudar encuentra una acción `EXECUTED` sin
   verificar (el proceso murió entre ambas), solo la verifica.
2. Si no, observa (`agent/observers.py`, reutiliza `WindowsAutomation`), actualiza la evidencia de
   requisitos (`agent/completion.py`), construye un contexto acotado y redactado
   (`agent/context.py`) y pide una decisión tipada (`planning/base.py`: `Decision`).
3. Aplica la decisión: `next_action` → nuevas acciones (ids asignados por Jarvis; el resto de un
   lote es provisional); `done` → **afirmación** que `CompletionChecker` acepta solo con evidencia;
   `stop` → fin seguro (`FAILED` si la última acción falló, `STOPPED` si no).

**Requisitos** (`agent/requirements.py`): solo se crean para aplicaciones de la allowlist
nombradas sin ambigüedad (abierta / en primer plano / cerrada). Se cumplen con una observación
fresca o con una acción `VERIFIED` cuyo objetivo resuelto es esa aplicación, independientemente de
las palabras usadas. Peticiones sin condiciones verificables completan cuando su última acción
tuvo éxito. Las referencias ("ese", "esa ventana", "ciérrala") solo se resuelven con pasos
anteriores de la misma petición (`AmbiguousReferenceError` si no).

**Límites y guardas** (`agent/limits.py`, aplicados por el runner, no por el planner): iteraciones,
acciones iniciadas, replanificaciones, tiempo global (`Task.elapsed_seconds` = ejecución +
planificación), ciclos (misma acción 3 veces, A→B→A→B, `done` rechazado dos veces) y repetición de
una acción no idempotente que ya falló. Fallos de política (DENY) y rechazos del usuario terminan la
tarea sin replanificar. `FallbackPlanner` no vuelve a consultar en la misma tarea un planner que ya
falló.

**Estado**: `TaskState.STOPPED` (terminal) para paradas seguras; `Task.progress` (fase,
iteraciones, acciones, replans, tiempo de planificación, motivo de parada) se persiste y se expone
en la API (`progress`) y la CLI. Fases: observing, planning, executing, replanning, waiting,
finished.

**Auditoría**: `agent.requirements`, `agent.observation`, `agent.plan`, `agent.action_selected`,
`agent.verification`, `agent.requirement_met`, `agent.replan`, `agent.limit_reached`,
`agent.cycle_detected`, `agent.completed`, `agent.stopped`, además de los eventos de seguridad de
v0.4 (`policy.evaluated`, `confirmation.*`, `action.*`). Solo hechos operacionales: el `rationale`
del planner es un resumen corto, nunca razonamiento.

## Resolución de objetivo (v0.4)

Entre la selección y la autorización, una herramienta puede resolver su **objetivo concreto** con
`resolve_target(args, ctx)`: una consulta de solo lectura (qué ejecutable de la allowlist, qué
ventana exacta). El executor:

1. la ejecuta con el mismo timeout que las herramientas;
2. guarda el resultado en `ActionRecord.target` y lo audita (`action.resolved`, con `requested` y
   `resolved` por separado; un fallo es `action.resolution_failed` y la acción no se ejecuta);
3. lo incluye en la `ConfirmationRequest` (`target`), que la CLI/API muestran como "objetivo";
4. lo incluye en el **hash de la aprobación** (`Action.digest(target)`): si al reanudar el objetivo
   resuelto es otro (otra ventana, título distinto), la aprobación anterior no vale y se vuelve a
   preguntar;
5. se lo pasa a `run` y `verify` como `ToolContext.target`: la herramienta actúa sobre lo resuelto,
   no sobre una nueva búsqueda.

Sin resolución (herramientas de archivos) el hash es idéntico al de versiones anteriores.

## Automatización de Windows (v0.4)

```text
WindowsTool (tools/builtin/windows.py)
   │  resolve_target / run / verify
   ▼
WindowsAutomation (desktop/automation.py)  ← interfaz; FakeWindowsAutomation en tests
   ▼
UiaWindowsAutomation (desktop/uia.py)      ← único módulo que importa pywinauto (perezoso)
   ├─ acciones: UI Automation (pywinauto): focus = set_focus, close = WindowPattern.Close
   ├─ lanzar: subprocess sin shell, ruta absoluta de la allowlist, sin argumentos
   └─ consultas: Win32 vía ctypes (EnumWindows, GetForegroundWindow, IsWindow, DWM cloaked)
```

- **Allowlist** (`desktop/catalog.py`): nombre pedido → `ApplicationEntry` (ejecutable, procesos y
  patrón de título que identifican sus ventanas). Entradas por defecto: Notepad y Calculadora.
  Configurable con `windows_applications`; los intérpretes/hosts de scripts están prohibidos
  incluso por configuración, y el lanzador vuelve a comprobarlo (defensa en profundidad).
- **Nombre pedido**: patrón estricto (palabras alfanuméricas separadas por un espacio, ≤ 40
  caracteres): sin rutas, comillas, operadores ni flags. Se valida en el schema, antes de
  planificar.
- **Identidad**: una ventana pertenece a una aplicación si su proceso está en
  `process_names` y, cuando el proceso es compartido (p. ej. `ApplicationFrameHost.exe` de las apps
  de la Store), su título encaja con `title_pattern`.
- **Ambigüedad**: si varias ventanas encajan, `ErrorKind.AMBIGUOUS` con las candidatas; nunca se
  elige una.
- **Cerrar**: solo ventanas de aplicaciones de la allowlist, categoría `close_application` con piso
  `REQUIRE_CONFIRMATION` (un grant `allow` no lo relaja), cierre normal por UI Automation. Si la
  ventana sigue (p. ej. diálogo "¿Guardar cambios?"), la verificación falla y no se fuerza nada.
  `unsaved_changes_suspected` (título que empieza por `*`) se muestra al confirmar como pista.
- **Antes de actuar**, `run` comprueba que el handle resuelto sigue existiendo y pertenece al mismo
  proceso (handles reutilizados).

### Verificación

| Acción | Ejecución (UIA / proceso) | Verificación (consulta Win32 nueva) |
|---|---|---|
| Abrir | lanzar el `.exe` de la allowlist y esperar una ventana | existe ventana de la app; `new_window` indica si es nueva o reutilizada |
| Enfocar | `set_focus` sobre el handle resuelto | `GetForegroundWindow()` es ese handle |
| Cerrar | `WindowPattern.Close` sobre el handle resuelto | `IsWindow(handle)` ya no es una ventana visible |

### Riesgos y límites

- **Timeouts e hilos abandonados**: toda llamada (resolución, acción, verificación) corre bajo el
  timeout de herramientas en un hilo daemon. Si UI Automation se cuelga (aplicación que no
  responde), Jarvis continúa pero el hilo queda abandonado hasta que termina o acaba el proceso;
  podría completar la acción tarde. Abrir y cerrar no son idempotentes: tras un timeout **no se
  reintentan** (resultado marcado `outcome_unknown`); enfocar sí (es idempotente).
- **Trabajo sin guardar**: el cierre siempre se confirma y nunca se fuerza, pero la pista de
  cambios sin guardar es heurística (asterisco en el título); muchas aplicaciones no la muestran.
  Notepad de Windows 11 es una ventana con pestañas: cerrar esa ventana afecta a todas sus
  pestañas.
- **Ambigüedad y TOCTOU**: el objetivo aprobado se vuelve a resolver y comparar al reanudar; aun
  así, entre la resolución y la acción hay una ventana de milisegundos (mitigada comprobando handle
  y proceso justo antes de actuar).
- **Foco**: Windows puede negar el cambio de primer plano a procesos en segundo plano; la
  verificación lo detecta y lo informa como fallo.

### Validación real (v0.4)

En Windows 11 (build 26200), con la Calculadora (sin ninguna abierta previamente):

- Pipeline en proceso: abrir (1,0 s, ventana nueva "Calculator" de `ApplicationFrameHost.exe`
  verificada) → listar → enfocar desde otra ventana (verificado por `GetForegroundWindow`) →
  cerrar con confirmación (verificado: no queda ventana) → `abre powershell` / `abre cmd`
  (rechazados al resolver) / `abre cmd.exe /c calc` (rechazado al planificar): 0 ventanas nuevas.
- `jarvis server` + CLI: el mismo ciclo por HTTP con el token (401 sin token, 403 con uno
  incorrecto), confirmación mostrando la ventana exacta y aprobación desde otro proceso.

## Token local de la API (v0.4)

`jarvis server` genera un token (`secrets.token_urlsafe(32)`) tras adquirir el lock de instancia y
lo escribe atómicamente en `<data_dir>/api_token` (0600 en POSIX; en Windows hereda el ACL de la
carpeta); lo borra al parar (solo si sigue siendo el suyo). Un middleware exige
`Authorization: Bearer <token>` (comparación en tiempo constante) en todo excepto `/health`:
401 sin token, 403 con uno incorrecto. La CLI lo lee del mismo directorio de datos. El middleware
de origen local (Host/Origin/JSON) se aplica antes que el del token.

Arranque automático (futuro): el runtime no depende del directorio actual si `JARVIS_DATA_DIR` y
`JARVIS_WORKSPACE_DIR` son absolutos, así que un acceso directo en Inicio o una tarea programada
podría lanzar `jarvis server` sin cambios.

## Proceso persistente (v0.3)

```text
CLI ──(urllib, loopback)──▶ FastAPI app ──▶ JarvisRuntime ──▶ Jarvis (fachada) ──▶ Orchestrator
                          (sin estado)      │                                       │
                                            ├─ TaskWorker: 1 hilo, cola de task_id  ├─ TaskRunner → ActionExecutor
                                            ├─ InstanceLock (jarvis.lock)           └─ StateStore (SQLite) ◀─ única fuente de verdad
                                            └─ recuperación al arrancar
```

### Ciclo de vida

```text
START → cargar config → construir runtime (LiteLLM se importa aquí, una vez)
      → adquirir lock de instancia → recuperar tareas → arrancar worker → servir API
      → señal (Ctrl+C / Ctrl+Break) → dejar de aceptar trabajo
      → la tarea en curso llega a un punto seguro (máx. shutdown_grace_seconds)
      → evento runtime.stopped → liberar lock → STOP
```

No hay estado en memoria que volcar: cada cambio de estado se confirma en SQLite y cada evento de
auditoría se escribe y vacía en el momento. Lo que quede en la cola al parar no se pierde: sigue en
SQLite como `PENDING`/`RUNNING` y se recupera al volver a arrancar.

### Ciclo de vida de los objetos

| Ámbito | Componentes |
|---|---|
| Proceso (uno por runtime) | `Settings`, `Workspace`, `ToolRegistry`, `PolicyEngine`, `StateStore`, `AuditSink`, planner (+ `LiteLLMProvider`), `ActionExecutor`, `TaskRunner`, `Orchestrator`, `Jarvis`, `JarvisRuntime`, `TaskWorker`, `InstanceLock`, app FastAPI |
| Por petición / ejecución | `Request`, `Task` (cargada de SQLite), `ExecutionContext` (reconstruido en cada ejecución), `Response`, vistas públicas |
| Persistente | SQLite (`state.db`), `audit.jsonl`, `jarvis.lock` |
| Sin estado | `PlanValidator`, `RequestGuard`, `ResponseBuilder`, reglas de política, middleware local |

No hay globales: `create_app(runtime)` recibe el runtime como objeto y lo guarda en el cierre de
los endpoints.

### Invariantes

- **Solo el worker ejecuta tareas** dentro de un runtime, una a la vez. Los endpoints solo crean
  tareas, registran decisiones o cancelan, siempre a través del dominio, y encolan el `task_id`.
- **Un runtime por directorio de datos** (lock de SO): ni dos servidores, ni un servidor y una
  CLI en modo local ejecutando a la vez. Así la recuperación de un servidor nunca reanuda una
  tarea que otro proceso aún está ejecutando.
- **SQLite es la única fuente de verdad**: la API lee siempre del `StateStore`; la cola del worker
  solo contiene identificadores.
- **Cada lectura es una instantánea**: una vista se construye a partir de una sola carga de la
  tarea (`snapshot`). Construirla con dos lecturas podía mezclar dos versiones de una tarea que el
  worker estaba modificando (encontrado y corregido durante v0.3).

### Tareas asíncronas y confirmaciones

`Orchestrator.submit()` crea la tarea `PENDING` y responde; `Orchestrator.process()` la planifica
y ejecuta en el worker. `approve`/`reject` llaman a `record_decision()`, que solo persiste la
decisión (con concurrencia optimista y reintento), y encolan la reanudación. Al reanudar, el
executor reevalúa la política y acepta la aprobación solo si no está consumida y su hash coincide
con los argumentos actuales: la API no contiene lógica de permisos. Mientras una decisión está
registrada pero aún no aplicada, la vista la expone (`decision`) para que los clientes sepan que la
tarea está a punto de continuar.

### Cancelación

Una tarea `PENDING` o `WAITING_PERMISSION` pasa a `CANCELLED` y no se ejecuta más. Si está
`RUNNING`, se marca `CANCELLED` con `details.cancelled_while = "running"`: la acción en curso **no
se interrumpe ni se revierte**. El runner lo detecta en su siguiente checkpoint (conflicto de
versión), no ejecuta más acciones y fusiona en el estado cancelado lo que realmente ocurrió (p. ej.
la acción en curso queda `executed`), emitiendo `task.cancel_observed`.

### Recuperación al arrancar

| Estado encontrado | Qué se hace |
|---|---|
| `PENDING` | Se procesa (nunca empezó a ejecutarse). |
| `RUNNING` | Se reanuda con las reglas existentes: un intento `STARTED` se marca `INTERRUPTED`; se reintenta solo si la herramienta es idempotente; si no, la acción falla sin repetirse. |
| `WAITING_PERMISSION` | Se deja esperando la decisión del usuario. |

Cada tarea recuperada genera `task.recovered` con el resultado. Si un hilo abandonado de un
runtime anterior (mismo proceso) intenta escribir después, el conflicto de versión le impide
sobrescribir el estado recuperado.

### Parada en un punto seguro

El runner consulta `should_stop` entre acciones: al parar, la acción en curso termina (dentro del
plazo de gracia), no empieza la siguiente y la tarea queda `RUNNING` para la recuperación. Si la
acción no termina dentro del plazo, el proceso sale igualmente (el worker es un hilo daemon) y la
recuperación aplica las reglas de resultado desconocido.

### Seguridad de la API local

`server_host` solo admite loopback (validado en `Settings`). Como un navegador puede enviar
peticiones a `127.0.0.1`, un middleware exige `Host` loopback (DNS rebinding), rechaza `Origin` no
locales (peticiones cruzadas) y exige `application/json` en métodos que modifican (evita formularios
sin preflight CORS). No hay autenticación: cualquier proceso del propio equipo puede usar la API.
El cliente de la CLI desactiva proxies para no enviar peticiones locales a un proxy configurado.

## Componentes

| Paquete | Responsabilidad |
|---|---|
| `core` | Modelo de dominio sin dependencias internas (más `timeouts.run_with_timeout`): `Request`, `Response`, `Task`/`TaskState` (máquina de estados explícita), `Action`, `ActionRecord`, `ActionResult`, `Failure`/`ErrorKind`, `Decision`/`PolicyVerdict`, `ActionCategory`. Todo serializable con Pydantic. |
| `context` | `ExecutionContext` (usuario, objetivo, tarea, aplicación, permisos, ítems relevantes, estado, datos efímeros) y `ContextProvider`, el punto de extensión para percepción y memoria. Hoy existe solo `SystemContextProvider` (hora, plataforma). |
| `permissions` | `PermissionSet`: scopes (`fs.write`) → `allow`/`confirm`/`deny`, comodines `prefijo.*`, denegación por defecto. |
| `safety` | `PolicyEngine` determinista con reglas componibles (`PermissionGrantRule`, `CategoryFloorRule`). Más restrictivo gana; si todas se abstienen → DENY; si una regla lanza excepción → DENY. Pisos por categoría (borrar, enviar mensajes, compras, transferencias, credenciales, config. del sistema → confirmación; `DANGEROUS` → DENY) que la configuración solo puede endurecer. `RequestGuard` valida la solicitud antes de planificar. |
| `desktop` | Interfaz `WindowsAutomation`, allowlist (`ApplicationCatalog`) y backend UI Automation (`uia.py`, solo Windows). |
| `tools` | Contrato `Tool` + `ToolSpec` inmutable (nombre, descripción, schema de argumentos, permisos, categorías, efectos secundarios, idempotencia, metadata). `ToolRegistry` impone invariantes al registrar (permisos y categorías obligatorios, argumentos `extra="forbid"`). `Workspace` confina rutas. Herramientas incluidas: `fs.write_file` (solo crea), `fs.overwrite_file` (categorías write+delete → confirmación), `fs.read_file`, `fs.list_dir`, `fs.delete_file`. `Workspace` rechaza absolutas, `..`, UNC, nombres de dispositivo de Windows, `:` (ADS) y nombres terminados en punto/espacio. |
| `planning` | Contrato `Planner` → `Plan` (acciones con `id` y `depends_on`, intentos fallidos, metadata operacional). `RuleBasedPlanner` (gramática explícita ES/EN). `ModelPlanner` (salida estructurada vía `ModelRouter`, timeout duro, límites de tamaño; los ids y dependencias los asigna Jarvis). `FallbackPlanner` (usa el primer planner cuyo plan **valida**; registra cada intento fallido). `PlanValidator` (herramientas conocidas, argumentos válidos, dependencias hacia atrás, límite de pasos). |
| `models` | `ModelProvider` (`chat`, `structured_output`, `capabilities`), `ModelRouter`, `parse_structured` (exactamente un objeto JSON, opcionalmente en un bloque ```` ``` ````; todo lo demás se rechaza), `LiteLLMProvider` (único módulo que importa LiteLLM, de forma perezosa; dependencia opcional `llm`) y `create_model_provider` (desde settings). |
| `execution` | `ToolSelector`, `ActionExecutor`, `RetryPolicy` + `classify_exception`, `ConfirmationHandler`, `VerificationService` (verificadores enchufables), `TaskRunner` (transiciones de tarea, cancelación, saltos). |
| `state` | `StateStore` con concurrencia optimista (`version`). `SqliteStateStore` (stdlib, migraciones SQL numeradas en `state/migrations/`) e `InMemoryStateStore`. |
| `observability` | `AuditEvent`/`Auditor` + sinks (`JsonlAuditLog`, `InMemoryAuditLog`), `redact()` para secretos y textos largos, y `Tracer` (no-op por defecto) como costura para OpenTelemetry. |
| `orchestration` | `Orchestrator` (coordina, no decide; `submit`/`process`/`record_decision`/`cancel`/`recover`), `ResponseBuilder`, fachada `Jarvis`. |
| `runtime` | `JarvisRuntime` (ciclo de vida del proceso), `TaskWorker` (hilo único), `InstanceLock`. |
| `interface` | `views` (contrato público, sin FastAPI), `api/` (app FastAPI, middleware local, `jarvis server`), `client` (backends HTTP y local para la CLI), CLI (`jarvis`). |
| `bootstrap.py` | Raíz de composición: único lugar donde se eligen implementaciones concretas. |

## Garantías de seguridad (cubiertas por tests)

- `Tool.run` solo se invoca desde `ActionExecutor`, y solo tras un veredicto ALLOW o una
  aprobación del usuario **vinculada al hash exacto de herramienta + argumentos** y de un solo uso.
- Una acción DENY nunca se ejecuta. REQUIRE_CONFIRMATION nunca se ejecuta en silencio: el
  manejador por defecto difiere y la tarea queda en `WAITING_PERMISSION`.
- La política se evalúa contra el `ToolSpec` registrado, nunca contra lo que diga el planner.
- Al reanudar se reevalúa la política con los permisos vigentes (una revocación se respeta).
- `ToolContext` no expone registro, ejecutor ni política: una herramienta no puede invocar a otra.
- Las rutas se confinan al workspace (`..`, absolutas y unidades se rechazan).
- La auditoría redacta claves sensibles (`password`, `token`, `api_key`, …) y trunca textos.
- No se solicita ni almacena razonamiento del modelo; solo plan, decisiones, acciones y resultados.

## Modelo de lenguaje: autoridad cero

El modelo solo puede producir un `StepProposal` (o el `PlanProposal` pre-v0.5; JSON estricto,
`extra="forbid"`). Caminos
revisados:

| Camino | Por qué no es posible |
|---|---|
| modelo → herramienta | El modelo no tiene ningún handle; sus acciones son datos que pasan por `PlanValidator` y luego por `ActionExecutor`, único llamador de `Tool.run`. |
| modelo → permisos | `Action` no tiene campos de permisos; los grants salen de settings y la política evalúa el `ToolSpec` registrado. Campos extra en el plan o en los argumentos se rechazan. |
| modelo → confirmación | Las aprobaciones solo se crean por el `ConfirmationHandler` o `Jarvis.approve`, ligadas al hash de los argumentos. El texto mostrado al usuario sale del `ToolSpec`, no de la descripción del modelo, y siempre se muestran herramienta y argumentos. |
| modelo → estado / límites | El modelo no escribe estado: el orquestador copia solo acciones validadas, la respuesta textual y metadata generada por Jarvis. Límites, reintentos y timeouts vienen de settings. |
| modelo → sandbox | Las rutas pasan por `Workspace` dentro de la herramienta (defensa independiente de la política). |
| modelo → DoS | Respuesta ≤ 100 000 caracteres, ≤ 50 acciones propuestas, `max_plan_steps`, timeout duro del modelo. |
| modelo → fallback | Forzar el fallback solo activa el planner determinista sobre el texto del propio usuario; un DENY en ejecución nunca replanifica. |

Hallazgo corregido en esta revisión: `fs.write_file` aceptaba `overwrite=true`, de modo que con
`fs.write=allow` un plan (p. ej. de un modelo manipulado) podía destruir contenido existente sin
confirmación. La sobrescritura es ahora una herramienta aparte con categoría `delete`.

## Timeouts

`run_with_timeout` ejecuta la llamada en un hilo *daemon* y devuelve el control al vencer el
plazo. Python no puede matar hilos: la llamada queda abandonada y su resultado se ignora.

- Herramienta: `tool_timeout_seconds` por intento → `ErrorKind.TIMEOUT` (con
  `outcome_unknown=True` si la herramienta tiene efectos secundarios). Se reintenta solo si es
  idempotente (o si la propia herramienta declaró el fallo reintentable).
- Tarea: `task_timeout_seconds` acota la suma persistida (`Task.execution_seconds`) de intentos y
  esperas de backoff; el tiempo esperando confirmación no cuenta. El backoff se recorta al
  presupuesto restante.
- Modelo: `model_timeout_seconds` se pasa a LiteLLM y además se impone como límite duro alrededor
  de la llamada, recortado al presupuesto restante de la tarea (v0.6); sin reintentos de LiteLLM
  ni del SDK. Al vencer, intento fallido de planificación (→ fallback).

## Recuperación de errores

`detectar → clasificar (ErrorKind) → registrar → ¿reintentar? → reintentar con límite → FAILED`.

- Reintentos solo para `TRANSIENT`/`TIMEOUT` marcados como reintentables, con backoff acotado.
- Límite por acción (`max_attempts_per_action`) y presupuesto global por tarea
  (`max_total_attempts_per_task`): no hay bucles infinitos.
- Herramientas **no idempotentes** solo se reintentan si la propia herramienta declaró el fallo
  reintentable (afirma que no hubo efecto).
- Antes de cada intento se persiste un `ActionResult` en estado `STARTED`. Si el proceso muere, al
  reanudar ese intento se marca `INTERRUPTED`; se reintenta solo si la herramienta es idempotente.
- Un fallo de verificación no se reintenta en esta versión.

## Estado y reanudación

La tarea completa (plan, registros de acción, intentos, veredictos, confirmaciones, verificación,
transiciones) se persiste como documento JSON con columnas indexadas. El contexto **no** se
persiste: se reconstruye en cada ejecución. La concurrencia optimista permite que `jarvis cancel`
desde otro proceso detenga una tarea en curso.

## Puntos de extensión (futuros, no implementados)

| Capacidad | Dónde encaja |
|---|---|
| Percepción (pantalla, ventana activa, cámara, micrófono) | `ContextProvider` |
| Memoria persistente / semántica | `ContextProvider` + nuevo paquete `memory` |
| Otros proveedores o varios modelos a la vez | Adaptadores de `ModelProvider` registrados en `ModelRouter` (hoy se registra uno) |
| Navegador (Playwright), MCP, más acciones de Windows | Nuevas `Tool` con sus categorías y `resolve_target`; verificadores externos vía `VerificationService` |
| PostgreSQL | Implementación de `StateStore` (el patrón `UPDATE … WHERE version = ?` ya está definido) |
| OpenTelemetry | Implementación de `Tracer`; sink de auditoría adicional |
| Streaming de eventos, UI | Nuevos endpoints/clientes sobre `JarvisRuntime` (la auditoría ya es la fuente) |
| Subagentes / ejecución paralela | `Plan.depends_on` ya modela el grafo; el `TaskRunner` actual es secuencial |

## Desviaciones respecto a la estructura propuesta

- No se crearon `cognition/`, `memory/`, `perception/`, `agents/` ni `migrations/` en la raíz:
  no tendrían código real todavía. Percepción y memoria encajan como `ContextProvider`.
- Las migraciones SQL viven en `src/jarvis/state/migrations/` para que se distribuyan con el
  paquete y el store las encuentre en tiempo de ejecución.
- `sandbox/` es el workspace por defecto de las herramientas de archivos (su contenido se ignora
  en git).
