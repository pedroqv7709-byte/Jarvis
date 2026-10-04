# Jarvis

Jarvis es un asistente personal de IA persistente. El objetivo a largo plazo es que perciba
contexto, razone, planifique, ejecute acciones con herramientas, verifique resultados, mantenga
estado y memoria, y opere siempre dentro de los permisos que define el usuario.

**Estado (v0.6)**: Jarvis ejecuta tareas de varios pasos con un **bucle agente**: observa,
decide el siguiente paso, lo valida, lo autoriza, lo ejecuta, lo verifica y vuelve a observar,
replanificando ante fallos dentro de límites duros. Corre como **proceso persistente local**
(`jarvis server`) con una **API solo en 127.0.0.1 protegida por token local**; la CLI es su
cliente. Trabaja con archivos confinados a un directorio y con **aplicaciones de Windows de una
allowlist** (abrir, enfocar, cerrar, listar), con verificación independiente. Las decisiones del
bucle puede tomarlas un **modelo de lenguaje real** vía LiteLLM (opcional, proveedor
intercambiable) con fallback determinista: el modelo propone cada paso, Jarvis valida, autoriza,
ejecuta y verifica. No hay voz, visión, memoria ni navegador (ver
[Qué NO está implementado](#qué-no-está-implementado)).

## Arquitectura

```text
CLI / futura UI ──HTTP 127.0.0.1──▶ API local (FastAPI) ──▶ JarvisRuntime (proceso persistente)
                                                             ├─ worker: una tarea a la vez
                                                             ├─ recuperación al arrancar
                                                             └─ Orchestrator ↓

request → context → request guard → planner ─┬─ ModelPlanner → ModelProvider → LiteLLM → proveedor
                                             └─ RuleBasedPlanner (fallback)
        → plan validator → task runner → policy (+ confirmación) → execution (timeout, reintentos)
        → verification → state → audit → response
```

El modelo **solo propone** un plan. Política, permisos, confirmaciones, límites, estado y
verificación son deterministas y viven fuera de cualquier modelo. Detalle en
[docs/architecture.md](docs/architecture.md).

## Requisitos

- Python 3.14
- [uv](https://docs.astral.sh/uv/) (si no está en el `PATH`, `python -m uv` también funciona)

## Instalación

```bash
git clone https://github.com/pedroqv7709-byte/Jarvis.git
cd Jarvis
uv sync                          # núcleo: planner determinista, archivos, servidor
uv sync --extra llm              # opcional: LiteLLM para planificar con un modelo
uv sync --extra windows          # opcional (solo Windows): automatización de aplicaciones
uv sync --extra llm --extra windows
```

Copia `.env.example` a `.env` para cambiar la configuración.

## Uso

### Proceso persistente (recomendado)

```bash
uv run jarvis server
```

Arranca Jarvis **una vez**: carga la configuración y las dependencias (incluido LiteLLM si está
configurado), recupera las tareas que quedaron sin terminar y escucha en `http://127.0.0.1:8765`
(solo local). Se detiene con **Ctrl+C** (o Ctrl+Break en Windows): termina la acción en curso si
cabe en `JARVIS_SHUTDOWN_GRACE_SECONDS` y cierra limpio. Lo que no termine se reanuda al volver a
arrancar.

Con el servidor en marcha, la CLI le envía las peticiones por HTTP en lugar de construir Jarvis en
cada comando:

```bash
uv run jarvis "crea un archivo de prueba"
uv run jarvis "crea el archivo notas.txt con el contenido \"hola mundo\" y luego léelo"
uv run jarvis "lista los archivos"
uv run jarvis "borra el archivo notas.txt"
uv run jarvis status        # modo (servidor/local) y planificador activo
```

**Si el servidor no está en marcha**, la CLI no levanta ninguno: ejecuta la petición dentro del
propio proceso (modo local, como antes de v0.3) y lo indica con un aviso. `--local` fuerza ese modo.
Un comando local que ejecuta o modifica tareas toma un lock sobre el directorio de datos, de modo
que nunca corre a la vez que un servidor (y un servidor no arranca mientras un comando local
trabaja).

Cada respuesta indica qué planificó la tarea: `Planificador: modelo`, `Planificador: determinista`
o `Planificador: determinista (fallback: el modelo no respondió a tiempo)`.

Borrar o sobrescribir un archivo requiere confirmación (piso de seguridad de la categoría
`delete`). En una terminal interactiva Jarvis pregunta `[s/N]`; con `--no-input` (o sin TTY) la
tarea queda en `waiting_permission` y se decide después, desde cualquier terminal o cliente:

```bash
uv run jarvis approve <task_id> a1
uv run jarvis reject <task_id> a1
```

Otros comandos:

```bash
uv run jarvis tasks [--state waiting_permission]   # listar tareas
uv run jarvis show <task_id>                       # estado completo de una tarea (JSON)
uv run jarvis audit <task_id>                      # auditoría de una tarea
uv run jarvis resume <task_id>                     # reanudar una tarea (solo modo local; el servidor reanuda solo)
uv run jarvis cancel <task_id>
uv run jarvis tools                                # herramientas, permisos y categorías
uv run jarvis --json run "lista los archivos"      # salida JSON
```

Datos generados (ignorados por git):

- `sandbox/`: único directorio que pueden tocar las herramientas de archivos.
- `.jarvis/state.db`: estado operacional (SQLite).
- `.jarvis/audit.jsonl`: auditoría estructurada (una línea JSON por evento).

## Bucle agente

Cada tarea avanza en ciclos, sin que el planner decida nunca qué está permitido:

```text
observar → decidir (planner) → validar → política/permisos → confirmación si toca
        → ejecutar → verificar → observar ... → completada | detenida | fallida
```

- **Observación**: antes de cada decisión Jarvis mira el escritorio (ventanas visibles y la que
  está en primer plano) con las mismas consultas de solo lectura que la verificación. Lo observado
  y lo que devuelven las herramientas es **dato no confiable**: llega al planner como datos, nunca
  como instrucciones.
- **Decisión**: el planner devuelve `next_action`, `done` o `stop` (estructura tipada; el modelo
  devuelve JSON estricto). Cada acción propuesta pasa de nuevo por validación, política,
  confirmación y verificación.
- **Replanificación**: si una acción falla (error, objetivo no encontrado, ambigüedad o
  verificación fallida), Jarvis observa el nuevo estado y pide otra decisión. El planner
  determinista solo reintenta una vez una acción idempotente cuya verificación falló; si no,
  se detiene explicando el motivo. Un **DENY** o un **rechazo tuyo terminan la tarea**: nunca se
  busca una vía alternativa.
- **Completion**: que el planner diga `done` no basta. Si la petición nombra una aplicación de la
  allowlist ("abre la calculadora y enfócala"), Jarvis exige evidencia de cada condición
  (observación o acción verificada sobre esa aplicación). Si no hay condiciones verificables
  (archivos, peticiones libres), exige que la última acción haya tenido éxito. Las referencias
  vagas ("enfoca ese") no se convierten en obligaciones inventadas: solo se resuelven con un paso
  anterior de la misma petición; si no, Jarvis se detiene y pide que la nombres.
- **Ya satisfecho**: si lo pedido ya se cumple (la calculadora ya está abierta), no se hace nada
  más y se completa con esa evidencia.
- **Resultado desconocido** (el proceso murió a mitad de una acción, o un timeout): una acción no
  idempotente nunca se repite a ciegas; Jarvis observa y decide, y si haría falta repetirla se
  detiene para que decidas tú.

**Límites duros** (configurables, fuera del modelo): iteraciones (`JARVIS_AGENT_MAX_ITERATIONS`,
20), acciones (`JARVIS_AGENT_MAX_ACTIONS`, 20), replanificaciones (`JARVIS_AGENT_MAX_REPLANS`, 5),
tiempo global por tarea (`JARVIS_TASK_TIMEOUT_SECONDS`, 300 s, incluye la planificación) y tamaño
del contexto (últimos 10 pasos, 30 ventanas, 12 000 caracteres). **Ciclos**: la misma acción
propuesta 3 veces, alternar A→B→A→B, o declarar `done` dos veces sin evidencia detiene la tarea.
Al alcanzar un límite la tarea queda en `STOPPED` con el motivo, visible en la API (`progress`) y
en la CLI ("Motivo de parada").

## API local

Disponible mientras corre `jarvis server`. Todas las respuestas son JSON; los `POST` deben
enviarse con `Content-Type: application/json` (también los que no llevan datos: `{}`).

| Método y ruta | Uso |
|---|---|
| `GET /health` | Proceso vivo, runtime listo, `StateStore` accesible, worker activo, planificador. No llama al modelo. 200 si todo está bien, 503 si no. |
| `POST /v1/tasks` `{"text": "..."}` | Crea una tarea. Responde `202` con `{"task_id", "status": "PENDING"}`; se procesa en segundo plano. |
| `GET /v1/tasks?status=&limit=` | Lista tareas (más recientes primero). |
| `GET /v1/tasks/{id}` | Estado, acciones, resultado, confirmaciones pendientes y error. |
| `POST /v1/tasks/{id}/approve` `{"action_id": "a1"}` | Aprueba una acción pendiente; la tarea se reanuda en segundo plano (`202`). |
| `POST /v1/tasks/{id}/reject` `{"action_id": "a1"}` | Rechaza una acción pendiente (`202`). |
| `POST /v1/tasks/{id}/cancel` `{}` | Cancela una tarea no terminada (`409` si ya terminó). |
| `GET /v1/tasks/{id}/events` | Eventos de auditoría de la tarea. |
| `GET /v1/tools` | Herramientas registradas, con permisos y categorías. |

Estados: `PENDING`, `RUNNING`, `WAITING_PERMISSION`, `COMPLETED`, `FAILED`, `CANCELLED`.

Todo excepto `/health` exige el token local (ver abajo). Ejemplos con `curl` desde bash (en
PowerShell usa `curl.exe` y `$(Get-Content .jarvis/api_token)`):

```bash
TOKEN=$(cat .jarvis/api_token)
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/v1/tasks -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"text": "lista los archivos"}'
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/v1/tasks/<task_id>
curl -X POST http://127.0.0.1:8765/v1/tasks/<task_id>/approve -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"action_id": "a1"}'
curl -X POST http://127.0.0.1:8765/v1/tasks/<task_id>/cancel -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{}'
```

**Aprobar por la API conserva todas las garantías**: la API solo entrega la decisión al dominio.
Al reanudar, la política se vuelve a evaluar con los permisos vigentes, y la aprobación vale una
sola vez y únicamente para los argumentos exactos que se mostraron. Un DENY sigue siendo DENY.

**Cancelar una tarea en ejecución** no interrumpe ni revierte la acción que esté corriendo en ese
momento: la tarea pasa a `CANCELLED`, no se ejecutan más acciones, y el estado registra lo que la
acción en curso llegó a hacer.

### Solo local y con token local

El servidor **solo escucha en loopback**: `JARVIS_SERVER_HOST` acepta únicamente `127.0.0.1`,
`::1` o `localhost`; cualquier otro valor (por ejemplo `0.0.0.0`) se rechaza al arrancar.

**Token local.** Cada vez que arranca, `jarvis server` genera un token aleatorio nuevo (256 bits)
y lo guarda en `<JARVIS_DATA_DIR>/api_token` (por defecto `.jarvis/api_token`, ignorado por git);
al apagarse lo borra. Todos los endpoints salvo `/health` exigen
`Authorization: Bearer <token>`: sin token responden `401`, con un token incorrecto `403`. La CLI
lo lee automáticamente, así que solo funciona si usa el mismo `JARVIS_DATA_DIR` que el servidor.
Solo quien pueda leer ese archivo puede usar la API: en Windows hereda los permisos de la carpeta
(dentro de tu perfil, por defecto no legible por otros usuarios); en Linux/macOS se crea con
permisos `0600`. No es un sistema de identidad: no hay cuentas, ni expiración, ni OAuth.

Como cualquier página web abierta en tu navegador puede enviar peticiones a `127.0.0.1`, la API
además rechaza `Host` no locales (DNS rebinding), `Origin` no locales (peticiones cruzadas desde
webs) y `POST` sin JSON (formularios que no pasan por preflight CORS).

## Automatización de Windows

Requiere Windows y `uv sync --extra windows` (instala `pywinauto`). Si falta cualquiera de las dos
cosas, o con `JARVIS_WINDOWS_AUTOMATION=off`, estas herramientas simplemente no se registran.

| Herramienta | Permiso | Categoría | Qué hace |
|---|---|---|---|
| `windows.get_windows` | `windows.read` | `read` | Lista las ventanas visibles (título, proceso, handle). |
| `windows.open_application` | `windows.open` | `open_application` | Abre una aplicación **de la allowlist** por nombre. |
| `windows.focus_window` | `windows.focus` | `window_management` | Trae al frente **exactamente una** ventana (por aplicación, título, proceso o handle). |
| `windows.close_application` | `windows.close` | `close_application` | Pide a **una** ventana de una aplicación de la allowlist que se cierre normalmente. **Siempre pide confirmación.** |

Frases del planner determinista: `abre la calculadora`, `abre el bloc de notas`, `enfoca la
calculadora`, `trae la ventana 'Título' al frente`, `cierra la calculadora`, `lista las ventanas`,
y encadenadas: `abre la calculadora y luego enfócala y luego ciérrala`. Con un modelo configurado,
el modelo puede proponer las mismas herramientas con las mismas validaciones.

**Allowlist.** Solo se abren aplicaciones conocidas: `notepad` (alias `bloc de notas`) y
`calculator` (alias `calculadora`, `calc`), resueltas a `%SystemRoot%\System32\notepad.exe` y
`calc.exe`. Puedes añadir más con `JARVIS_WINDOWS_APPLICATIONS` (ver `.env.example`): ruta
absoluta a un `.exe`, sin argumentos. Intérpretes y ejecutores de scripts (`cmd`, `powershell`,
`pwsh`, `wscript`, `cscript`, `mshta`, `rundll32`, `regsvr32`, `python`…) **nunca** se pueden
añadir. El nombre que llega en la petición solo sirve para buscar en la allowlist: no puede
contener rutas, comillas, operadores de shell ni flags (`cmd.exe /c …` se rechaza antes de
planificar), y un nombre desconocido no ejecuta nada.

**Objetivo resuelto y confirmación.** Antes de autorizar, cada herramienta resuelve su objetivo
concreto (qué ejecutable, qué ventana exacta) sin modificar nada. Ese objetivo se audita aparte de
lo que se pidió, se muestra al confirmar (`objetivo: ventana '*informe.txt - Bloc de notas'
(Notepad.exe); posibles cambios sin guardar`) y forma parte de la aprobación: si la ventana cambia
(otra ventana, nuevo título con `*`) antes de ejecutar, hay que volver a confirmar. Si varias
ventanas encajan con los criterios, Jarvis **no elige**: devuelve la lista de candidatas
(`ambiguous`) para que se precise, por ejemplo con `handle`.

**Verificación independiente.** Las acciones van por UI Automation; la verificación vuelve a
consultar el sistema por otro camino (Win32): abrir → existe una ventana de esa aplicación (e
indica si es nueva o una existente reutilizada); enfocar → la ventana en primer plano es la
pedida; cerrar → la ventana ya no existe. Si la aplicación pide guardar cambios, la ventana sigue
abierta, la verificación falla y **no se fuerza nada** (nunca se usa `taskkill` ni se termina un
proceso).

**Límites conocidos.** En Windows 11 Notepad usa una sola ventana con pestañas: abrirlo puede
añadir una pestaña a una ventana existente (se informa como "reutilizada") y cerrarlo cierra esa
ventana entera, con todas sus pestañas, tras tu confirmación. Una operación de UI que se cuelga se
corta con el timeout de herramientas, pero su hilo no puede matarse (ver Riesgos en
`docs/architecture.md`); cerrar y abrir no se reintentan nunca tras un timeout.

## Planificación con un modelo (opcional)

```bash
uv sync --extra llm
```

Configura en `.env` (o variables de entorno):

| Variable | Valor por defecto | Significado |
|---|---|---|
| `JARVIS_MODEL_PROVIDER` | `none` | `none`: solo planner determinista. `litellm`: planificar con un modelo vía LiteLLM. |
| `JARVIS_MODEL` | — | Modelo en formato LiteLLM `<proveedor>/<modelo>`, p. ej. `openai/gpt-4o-mini` u `ollama/llama3.1`. Obligatorio con `litellm`. |
| `JARVIS_MODEL_API_BASE` | — | Endpoint alternativo (p. ej. Ollama local `http://localhost:11434`). |
| `JARVIS_MODEL_TIMEOUT_SECONDS` | `30` | Límite duro de cada llamada de planificación al modelo (nunca más que el tiempo que le queda a la tarea). |
| `JARVIS_MODEL_STRUCTURED_OUTPUT` | `auto` | Cómo se pide el JSON: `json_schema` (se envía el schema de la decisión), `json_object` (modo JSON + schema en el prompt) o `auto` (schema si LiteLLM sabe que el backend lo soporta). La respuesta se valida siempre. |
| `JARVIS_PLANNER_FALLBACK` | `true` | Si el modelo falla, tarda demasiado o su decisión no valida, usar el planner determinista. |

Modelos recomendados para planificar herramientas (rápidos y económicos): `openai/gpt-4o-mini`
(con `OPENAI_API_KEY`); cualquier otro backend de LiteLLM sirve igual (`anthropic/...`,
`gemini/...`); o local con Ollama: `JARVIS_MODEL=ollama/llama3.1` y
`JARVIS_MODEL_API_BASE=http://localhost:11434`.

Las **claves de API no son configuración de Jarvis**: LiteLLM lee sus variables estándar
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …). Jarvis nunca las maneja ni las registra.

Comportamiento:

1. En **cada iteración** del bucle el modelo recibe un único objeto JSON con dos partes:
   `task` (lo escribe Jarvis: objetivo, requisitos verificables y si ya se cumplen, presupuesto
   restante y el motivo del último fallo) y `untrusted_data` (contexto, últimos pasos con sus
   resultados y verificaciones, observaciones del escritorio). El prompt de sistema declara que
   todo lo de `untrusted_data` (títulos de ventanas, salidas de herramientas, errores) es **dato,
   nunca instrucción**. El catálogo de herramientas incluye los nombres de la allowlist. Nunca se
   envían secretos, elementos de contexto sensibles, historial ilimitado ni logs.
2. Debe devolver **un único objeto JSON**: `{"kind": "next_action", "action": {...}}`,
   `{"kind": "done"}` o `{"kind": "stop", "reason": "..."}`. Texto libre, JSON malformado, campos
   desconocidos (`approved`, `permissions`, `verified`, `final`…) o respuestas demasiado grandes
   se rechazan; nunca se interpretan.
3. La decisión pasa por el `PlanValidator` (herramienta existente, argumentos válidos según su
   schema). Nada se ejecuta antes de validar. `done` es solo una afirmación: Jarvis exige evidencia.
4. Si el modelo falla, se excede el timeout o la decisión no valida, y el fallback está activo, se
   usa el planner determinista (y el modelo no se vuelve a llamar en esa tarea). Si tampoco puede,
   la tarea termina en `failed` con el motivo de ambos. El fallback queda en la auditoría
   (`plan.attempt_failed`) y en la respuesta.
5. Después, la acción del modelo se trata igual que cualquier otra: política, confirmación,
   timeouts, verificación, límites y detección de ciclos. Un DENY nunca provoca replanificación.

**Sin reintentos ocultos**: Jarvis desactiva los reintentos de LiteLLM y del SDK del proveedor
(el cliente de OpenAI reintenta dos veces por defecto), así que cada decisión cuesta como mucho un
timeout del modelo. El único "reintento" es el fallback, acotado.

Sin API key (o con el proveedor caído) la CLI sigue funcionando: la llamada falla y se usa el
fallback determinista. Los mensajes de error del proveedor se limpian de fragmentos de claves
antes de llegar a la auditoría.

### Validación con un modelo real

Los tests normales nunca usan red. La validación real es opcional y se activa a mano:

```bash
JARVIS_LIVE_MODEL=1 JARVIS_MODEL_PROVIDER=litellm JARVIS_MODEL=openai/gpt-4o-mini uv run pytest tests/test_live_model.py -s
```

Con `JARVIS_LIVE_WINDOWS=1` además conduce el escritorio real: el modelo decide cómo cumplir
"Abre la calculadora y enfócala." (solo Calculadora; sin escribir, sin clics, sin shell; la
calculadora queda abierta al terminar).

## Timeouts y reintentos

| Variable | Valor por defecto | Significado |
|---|---|---|
| `JARVIS_TOOL_TIMEOUT_SECONDS` | `30` | Límite duro por intento de herramienta. |
| `JARVIS_TASK_TIMEOUT_SECONDS` | `300` | Tiempo total de ejecución de herramientas + esperas de reintento por tarea (no cuenta el tiempo esperando tu confirmación). |
| `JARVIS_MAX_ATTEMPTS_PER_ACTION` | `3` | Intentos máximos por acción. |
| `JARVIS_MAX_TOTAL_ATTEMPTS_PER_TASK` | `20` | Intentos máximos por tarea. |

Solo se reintentan fallos transitorios o timeouts, con backoff acotado. Una herramienta **no
idempotente** que excede su timeout **no se reintenta** (no se sabe si llegó a ejecutarse): la
acción falla con `outcome_unknown`.

## Servidor

| Variable | Valor por defecto | Significado |
|---|---|---|
| `JARVIS_SERVER_HOST` | `127.0.0.1` | Solo direcciones loopback (`127.0.0.1`, `::1`, `localhost`). |
| `JARVIS_SERVER_PORT` | `8765` | Puerto local de la API (1024–65535). |
| `JARVIS_SHUTDOWN_GRACE_SECONDS` | `10` | Al detener, cuánto esperar a que la tarea en curso llegue a un punto seguro. |

Al arrancar, el servidor revisa SQLite: las tareas `PENDING` se procesan; las que estaban
`RUNNING` se reanudan con las reglas de recuperación (un intento a medias de una herramienta no
idempotente **no se repite**: la acción falla como `interrupted`); las `WAITING_PERMISSION` siguen
esperando tu decisión. Cada caso queda en la auditoría (`task.recovered`).

## Gramática del planner determinista

Entiende solo lo siguiente (español e inglés); ante cualquier otra cosa falla explícitamente:

| Intención | Ejemplos |
|---|---|
| Crear archivo | `crea un archivo de prueba`, `crea el archivo a.txt con el contenido "hola"`, `create file a.txt with content hi` |
| Sobrescribir (pide confirmación) | `sobrescribe el archivo a.txt con el contenido nuevo` |
| Leer | `lee el archivo a.txt`, `muestra el contenido del archivo a.txt`, `léelo` |
| Listar | `lista los archivos`, `lista los archivos en docs` |
| Borrar (pide confirmación) | `borra el archivo a.txt`, `bórralo` |
| Abrir aplicación | `abre la calculadora`, `abre el bloc de notas`, `open notepad` |
| Enfocar ventana | `enfoca la calculadora`, `trae la ventana 'Título' al frente`, `enfócala` |
| Cerrar aplicación (pide confirmación) | `cierra la calculadora`, `ciérrala` |
| Listar ventanas | `lista las ventanas`, `qué ventanas hay abiertas` |

Varios pasos se encadenan con `;`, `y luego`, `y después` o `and then`. El texto entre comillas
nunca se divide.

## Tests y calidad

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Los tests de la API usan `TestClient` con el runtime real (sin abrir puertos); un único test
arranca uvicorn en un socket loopback real para probar la CLI como cliente. Los tests no usan
Internet ni API keys: el modelo se sustituye por respuestas grabadas en la frontera
de `litellm.completion` (por encima, todo es código real: adaptador, parseo, `ModelPlanner`,
fallback, política, verificación...). Algunos de esos "modelos" obedecen inyecciones a propósito
(`tests/test_model_loop.py`): la seguridad no depende de que el modelo se porte bien. Un test usa
la librería LiteLLM real en modo offline (`mock_response`) y otros
(`tests/test_litellm_wire.py`) usan LiteLLM y el SDK de OpenAI reales por HTTP contra un falso
endpoint en `127.0.0.1` (formato en el cable, sin reintentos, bucle completo); se omiten
(`skipped`) si no instalaste el extra `llm`. Los tests con modelo real (`tests/test_live_model.py`) se omiten salvo que los
actives.

## Estructura

```text
src/jarvis/
├── core/            dominio: Request, Response, Task/TaskState, Action, Failure, Decision, timeouts
├── context/         ExecutionContext, ContextBuilder, ContextProvider
├── permissions/     PermissionSet (scopes → allow/confirm/deny)
├── safety/          PolicyEngine, reglas, pisos por categoría, RequestGuard
├── tools/           contrato Tool/ToolSpec, ToolRegistry, Workspace, builtin/ (filesystem, windows)
├── desktop/         interfaz WindowsAutomation, allowlist de aplicaciones, backend UI Automation
├── planning/        Planner, PlanValidator, RuleBasedPlanner, ModelPlanner, FallbackPlanner
├── models/          ModelProvider, ModelRouter, parseo estructurado, adaptador LiteLLM
├── execution/       ToolSelector, ActionExecutor, RetryPolicy, confirmación, verificación, TaskRunner
├── state/           StateStore, SqliteStateStore (+ migraciones), InMemoryStateStore
├── observability/   auditoría, redacción, tracer (costura para OpenTelemetry)
├── orchestration/   Orchestrator, ResponseBuilder, fachada Jarvis
├── runtime/         proceso persistente: ciclo de vida, worker, lock de instancia
├── interface/       CLI, cliente HTTP/local, vistas públicas, api/ (FastAPI)
├── bootstrap.py     raíz de composición
└── config.py        Settings (variables JARVIS_*, .env)
tests/               unitarios, seguridad, modelo no confiable, timeouts, API, runtime e integración
docs/                arquitectura
sandbox/             workspace por defecto de las herramientas
```

## Principios arquitectónicos

1. **El LLM no es la autoridad.** Propone planes; política, permisos, estado y verificación son
   deterministas. Toda salida del modelo es dato no confiable.
2. **Denegación por defecto.** Un permiso no concedido se deniega; una regla que falla deniega.
3. **Separación estricta**: herramienta *seleccionada* ≠ *autorizada* ≠ *ejecutada* ≠ *verificada*.
4. **Confirmación vinculada**: una aprobación sirve una sola vez y solo para los argumentos exactos;
   el texto de la confirmación sale de la herramienta registrada, nunca del planner.
5. **Estado explícito y serializable**, con concurrencia optimista y reanudación tras fallos.
6. **Auditoría operacional, sin chain-of-thought**, con redacción de secretos.
7. **Componentes reemplazables** detrás de protocolos; el cableado está solo en `bootstrap.py`.
8. **Límites explícitos**: pasos, intentos, tiempo por herramienta, por tarea y por llamada al modelo.
9. **Objetivo explícito**: lo pedido, lo resuelto por Jarvis y lo que el sistema muestra se
   registran por separado; las aprobaciones se ligan al objetivo concreto.
10. **Una única fuente de verdad**: SQLite vía `StateStore`. La API y el runtime no guardan estado
   propio; solo el worker del runtime ejecuta tareas.

## Qué está implementado

- Modelo real en el bucle (v0.6): el modelo decide cada paso con salida estructurada (JSON
  Schema cuando el backend lo soporta), contexto acotado que separa los datos de Jarvis de los
  datos no confiables, timeout acotado al presupuesto de la tarea, sin reintentos ocultos,
  inyecciones en ventanas/salidas/contexto probadas con modelos que las obedecen.
- Bucle agente (v0.5): observación, decisiones incrementales, replanificación, completion por
  evidencia, límites duros, detección de ciclos, estado `STOPPED` y progreso en API/CLI.
- Automatización de Windows con allowlist: abrir, listar, enfocar y cerrar (sin forzar) con
  verificación independiente; sin ejecución arbitraria de comandos. Validado en un Windows real
  con la Calculadora (ver `docs/architecture.md`).
- Token local para la API (rotado en cada arranque), leído automáticamente por la CLI.
- Proceso persistente `jarvis server` con API local FastAPI (solo loopback, protecciones contra
  peticiones desde el navegador), tareas asíncronas con `task_id`, aprobación/rechazo/cancelación
  desde cualquier cliente, recuperación conservadora al arrancar, apagado limpio y lock de una
  instancia por directorio de datos.
- CLI como cliente del servidor, con modo local explícito y avisado cuando no hay servidor.
- Pipeline completo request → … → response, cubierto por tests de integración (con y sin modelo,
  por llamada directa y por HTTP).
- Adaptador LiteLLM real (`ModelProvider`), configurable por settings, como dependencia opcional.
- `ModelPlanner` con salida estructurada estricta, límites de tamaño y timeout duro.
- `FallbackPlanner`: modelo → determinista, auditado y visible en la respuesta.
- Timeouts duros por herramienta, presupuesto de tiempo por tarea, timeout del modelo.
- Máquina de estados de tarea con transiciones validadas y registradas.
- Motor de políticas determinista (ALLOW / DENY / REQUIRE_CONFIRMATION) con permisos por scope y
  pisos por categoría.
- Confirmación interactiva o diferida (`approve`/`reject`), vinculada al hash de los argumentos.
- Cinco herramientas de archivos confinadas a `sandbox/` (crear, sobrescribir, leer, listar,
  borrar), con verificación. El confinamiento rechaza rutas absolutas, `..`, UNC, nombres de
  dispositivo de Windows (`NUL`, `CON`…) y flujos alternativos (`a.txt:x`).
- Reintentos con clasificación de errores, backoff y respeto de idempotencia.
- Persistencia SQLite con migraciones, reanudación tras crash y cancelación desde otro proceso.
- Auditoría JSONL redactada. CLI.

## Qué NO está implementado

- Validación contra un proveedor real *en este entorno*: el camino está integrado y cubierto con
  respuestas grabadas y LiteLLM offline, y hay tests opt-in para un modelo real
  (`tests/test_live_model.py`), pero no se han ejecutado por falta de credenciales.
- Streaming de eventos (la API se consulta por polling). Identidades/cuentas: el token local es
  el único control de acceso.
- Automatización de Windows más allá de abrir/listar/enfocar/cerrar: escribir, hacer clic, leer
  el contenido de ventanas, integración específica con Teams u otras aplicaciones. Cierre forzado.
- Aclaraciones interactivas: ante una petición ambigua Jarvis se detiene con el motivo; no
  pregunta y espera respuesta dentro de la misma tarea.
- Ejecución paralela: el servidor acepta varias peticiones, pero ejecuta una tarea a la vez.
- PostgreSQL, Redis, Docker, OpenTelemetry real (solo la costura `Tracer`).
- Percepción (pantalla, cámara, micrófono), voz, visión, memoria persistente o semántica.
- Navegador / Playwright, automatización de Windows, MCP, Teams, correo, mensajería.
- Subagentes, ejecución paralela, rollback de acciones.
- Cancelación real de una herramienta colgada: al vencer el timeout (o al cancelar) Jarvis
  continúa, pero el hilo de la herramienta no puede matarse y queda abandonado hasta que termine
  el proceso.
- Arranque automático con Windows: `jarvis server` se inicia a mano. Para hacerlo en el futuro
  (Inicio de Windows o Programador de tareas) basta con lanzar `jarvis server` con
  `JARVIS_DATA_DIR` y `JARVIS_WORKSPACE_DIR` absolutos, porque los valores por defecto son
  relativos al directorio actual.
