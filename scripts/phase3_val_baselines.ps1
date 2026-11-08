# Fase 3 (piloto): sanity checks y baselines en VAL, 1 semilla, 10 pares x 200 batallas.
# BORRADOR asistido por IA. Requiere Showdown en el puerto 8000.
# Uso: .\scripts\phase3_val_baselines.ps1 -Bc results\bc_...\model.pt -Vhat results\vhat_...\model.pt
param([Parameter(Mandatory=$true)][string]$Bc, [Parameter(Mandatory=$true)][string]$Vhat, [int]$Seed = 0)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$py = ".\.venv\Scripts\python.exe"
# Sanity checks
& $py -m bringvgc.eval.run_arena --method usage --opponent usage --pairs val --seed $Seed
& $py -m bringvgc.eval.run_arena --method usage_argmax --mirror --pairs val --seed $Seed
# Baselines contra el rival Usage estocastico
foreach ($m in "random", "usage_argmax", "tipos") {
  & $py -m bringvgc.eval.run_arena --method $m --pairs val --seed $Seed
}
& $py -m bringvgc.eval.run_arena --method bc --bc-ckpt $Bc --pairs val --seed $Seed
& $py -m bringvgc.eval.run_arena --method planner --vhat-ckpt $Vhat --pairs val --seed $Seed
