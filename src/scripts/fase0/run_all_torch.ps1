# Reproduce todas las tablas de la Etapa 2 (plan v2 §9, §11). BORRADOR asistido por IA.
# Una linea por bloque/tabla. Requiere: .venv con requirements.txt, Showdown 913da36 en el puerto 8000
#   (cd ..\vgc-bench\pokemon-showdown; node pokemon-showdown start --no-security)
# y el log set (configs/data.yaml o la variable BRINGVGC_LOGS).
#
# Uso:
#   .\scripts\run_all.ps1 -Stage data      # Fase 1: dataset, splits, tests
#   .\scripts\run_all.ps1 -Stage offline   # Fase 2: rho_BC (3 semillas) y v-hat (1 semilla)
#   .\scripts\run_all.ps1 -Stage val       # Fase 3: sanity checks y baselines en val
#   .\scripts\run_all.ps1 -Stage rl        # Fase 4: RL 3 semillas x 50k (beta_H=0.01) + 3 x 15k (beta_H=0)
#   .\scripts\run_all.ps1 -Stage test      # Fase 5: UNA SOLA VEZ, despues de fijar docs/EXPERIMENTS.md
# Los checkpoints se pasan con -Bc0 -Bc1 -Bc2 -Vhat -Agent0 -Agent1 -Agent2 (rutas a model.pt / ckpt_best_val.pt).
param(
  [Parameter(Mandatory=$true)][ValidateSet("data","offline","val","rl","test")][string]$Stage,
  [string]$Bc0, [string]$Bc1, [string]$Bc2, [string]$Vhat,
  [string]$Agent0, [string]$Agent1, [string]$Agent2
)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$py = ".\.venv\Scripts\python.exe"
$Bcs = @($Bc0, $Bc1, $Bc2); $Agents = @($Agent0, $Agent1, $Agent2)

switch ($Stage) {
  "data" {
    & $py scripts/build_dataset.py --config configs/data.yaml          # tabla de estadisticas del dataset
    & $py -m pytest -q                                                 # tests (incluye leakage)
  }
  "offline" {
    foreach ($s in 0,1,2) { & $py -m bringvgc.train.bc --config configs/bc.yaml --seed $s }   # NLL por |C| en val
    & $py -m bringvgc.train.vhat --config configs/vhat.yaml --seed 0                           # acc/AUC de v-hat en val
    & $py -m bringvgc.eval.diagnostics --partition val --ckpts $Bc0 $Bc1 $Bc2                  # diagnosticos en val
  }
  "val" {
    & $py -m bringvgc.eval.run_arena --method usage --opponent usage --pairs val               # sanity: Usage vs Usage
    & $py -m bringvgc.eval.run_arena --method usage_argmax --mirror --pairs val                # sanity: mismo bring
    foreach ($s in 0,1,2) { & $py -m bringvgc.eval.run_arena --method random --pairs val --seed $s }
    & $py -m bringvgc.eval.run_arena --method usage_argmax --pairs val
    & $py -m bringvgc.eval.run_arena --method tipos --pairs val
    & $py -m bringvgc.eval.run_arena --method planner --vhat-ckpt $Vhat --pairs val
    foreach ($s in 0,1,2) { & $py -m bringvgc.eval.run_arena --method bc --bc-ckpt $Bcs[$s] --pairs val --seed $s }
    & $py -m bringvgc.eval.aggregate --pairs val --out results/tabla_val.md                   # tabla de baselines en val
  }
  "rl" {
    # Cambiar bc_ckpt en configs/rl.yaml por la semilla correspondiente antes de cada corrida,
    # o copiar la config. Reanudar una corrida cortada: --resume results/rl_<id>
    foreach ($s in 0,1,2) { & $py -m bringvgc.train.rl --config configs/rl.yaml --seed $s }
    foreach ($s in 0,1,2) { & $py -m bringvgc.train.rl --config configs/rl.yaml --seed $s --episodes 15000 --beta-H 0 }
  }
  "test" {
    # SOLO Fase 5, una vez, con la regla de docs/EXPERIMENTS.md ya escrita y commiteada.
    $n = 1000
    foreach ($s in 0,1,2) {
      & $py -m bringvgc.eval.run_arena --method agent --agent-ckpt $Agents[$s] --pairs test --allow-test --n-per-pair $n --seed $s
      & $py -m bringvgc.eval.run_arena --method bc --bc-ckpt $Bcs[$s] --pairs test --allow-test --n-per-pair $n --seed $s
      & $py -m bringvgc.eval.run_arena --method random --pairs test --allow-test --n-per-pair $n --seed $s
    }
    foreach ($m in "usage_argmax","tipos") { & $py -m bringvgc.eval.run_arena --method $m --pairs test --allow-test --n-per-pair $n }
    & $py -m bringvgc.eval.run_arena --method planner --vhat-ckpt $Vhat --pairs test --allow-test --n-per-pair $n
    & $py -m bringvgc.eval.aggregate --pairs test --reference usage_argmax --out results/tabla1_test.md   # Tabla 1
    & $py -m bringvgc.eval.aggregate --pairs test --reference bc --out results/tabla1_test_vs_bc.md       # agente vs BC
    & $py -m bringvgc.eval.diagnostics --partition test --allow-test --ckpts $Bc0 $Bc1 $Bc2              # Tabla 2
  }
}
