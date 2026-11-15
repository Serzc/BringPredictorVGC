# Limpieza pendiente (para José)

> Nota preparada con IA el 14-11-2026 y revisada contra el repo ese día. Ninguno de estos archivos está en git, así que borrarlos no cambia el historial ni hace falta commit.

## Qué borrar y por qué

| Ruta | Qué es | Por qué se puede borrar |
|---|---|---|
| `tests/` (raíz del repo) | Solo carpetas `__pycache__` | Los tests reales están en `src/tests/`. Esta carpeta quedó vacía después de la reestructuración |
| `src/bringvgc/` | Solo carpetas `__pycache__` | Era el paquete viejo. El código ahora vive directo en `src/` |
| `src/scripts/fase0/run_all_torch.ps1` | Runner de la era torch | El proyecto ya no usa torch. Todo se corre con `python src/main.py ...` |
| `src/scripts/fase0/phase3_val_baselines_torch.ps1` | Runner de la era torch | Mismo motivo |
| `%TEMP%\bringvgc_torchfree_venv` | Venv temporal de una prueba | El venv real es `.venv` en la raíz del repo |
| `%TEMP%\bringvgc_smoke_fix`, `%TEMP%\bringvgc_validate` | Carpetas temporales de pruebas | Ya no se usan |

**No borrar:** `src/output/legacy_torch/`, porque `docs/EXPERIMENTS.md` §7 la cita para la medición de $T$.

## Cómo hacerlo

Desde `cmd`, en la raíz del repo (`BringPredictorVGC`):

```cmd
rmdir /s /q tests
rmdir /s /q src\bringvgc
del src\scripts\fase0\run_all_torch.ps1 src\scripts\fase0\phase3_val_baselines_torch.ps1
rmdir /s /q "%TEMP%\bringvgc_torchfree_venv" "%TEMP%\bringvgc_smoke_fix" "%TEMP%\bringvgc_validate"
```

Si `src\scripts\fase0\` queda vacía, también se puede borrar con `rmdir src\scripts\fase0`.

## Comprobar que no se rompió nada

```cmd
.venv\Scripts\python -m pytest src/tests -q
```

Tiene que dar el mismo resultado que antes de borrar (los 20 tests de `test_parser_annotated.py` siguen saltándose hasta que se anoten los replays).
