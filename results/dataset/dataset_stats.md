# Estadisticas del dataset (generado por scripts/build_dataset.py)

Parseo: 6902/6905 validos (99.96%). Excluidos: {'teamsize_not_4': 3}.

| particion | batallas | unidades | series_bo3 | player_brings | desde | hasta | v2_% | v1_% | v0_% | mediana_turnos | cobertura_vocab |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 4831 | 3952 | 882 | 9662 | 2026-04-11 03:07 | 2026-05-25 16:51 | 84.1 | 12.1 | 3.8 | 7 | - |
| val | 1036 | 845 | 205 | 2072 | 2026-05-25 17:09 | 2026-06-05 01:50 | 83.6 | 12.0 | 4.3 | 7 | {'species': 0.9989, 'item': 0.9988, 'ability': 0.9998, 'move': 0.9999} |
| test | 1035 | 874 | 158 | 2070 | 2026-06-05 01:50 | 2026-06-20 00:37 | 84.9 | 11.4 | 3.7 | 7 | {'species': 0.9906, 'item': 0.9994, 'ability': 0.9956, 'move': 0.9996} |

Revelacion completa: ganadores 73.6%, perdedores 94.7% (nunca se usa como feature).

Pares congelados (ver splits/pairs_report.json):
```
{
 "val": {
  "candidate_pairs": 791,
  "pairs_with_0_seen_teams": 386,
  "pairs_with_1_seen_team": 335,
  "pairs_with_2_seen_teams": 70,
  "chosen": 10,
  "relaxed_to_unseen_pair": false,
  "chosen_seen_teams": [
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0
  ]
 },
 "test": {
  "candidate_pairs": 816,
  "pairs_with_0_seen_teams": 462,
  "pairs_with_1_seen_team": 305,
  "pairs_with_2_seen_teams": 49,
  "chosen": 20,
  "relaxed_to_unseen_pair": false,
  "chosen_seen_teams": [
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0
  ]
 },
 "validator": {
  "teams_checked": 12256,
  "teams_legal": 12167,
  "battles_with_illegal_team": 80,
  "battles_with_illegal_team_ids": [
   "gen9championsvgc2026regma-2597409761",
   "gen9championsvgc2026regma-2597381246",
   "gen9championsvgc2026regma-2596506955",
   "gen9championsvgc2026regma-2596517440",
   "gen9championsvgc2026regma-2580331834",
   "gen9championsvgc2026regma-2601968974",
   "gen9championsvgc2026regma-2607367469",
   "gen9championsvgc2026regma-2601956339",
   "gen9championsvgc2026regma-2607361565",
   "gen9championsvgc2026regma-2613838257",
   "gen9championsvgc2026regma-2613560793",
   "gen9championsvgc2026regma-2613829508",
   "gen9championsvgc2026regma-2613563946",
   "gen9championsvgc2026regma-2616649630",
   "gen9championsvgc2026regma-2617116328",
   "gen9championsvgc2026regma-2615414369",
   "gen9championsvgc2026regma-2616048400",
   "gen9championsvgc2026regma-2615856979",
   "gen9championsvgc2026regma-2615095001",
   "gen9championsvgc2026regma-2616051265",
   "gen9championsvgc2026regma-2615142101",
   "gen9championsvgc2026regma-2615042135",
   "gen9championsvgc2026regma-2615872715",
   "gen9championsvgc2026regma-2617063205",
   "gen9championsvgc2026regma-2616173809",
   "gen9championsvgc2026regma-2615412683",
   "gen9championsvgc2026regma-2615811298",
   "gen9championsvgc2026regma-2616043914",
   "gen9championsvgc2026regma-2615150456",
   "gen9championsvgc2026regma-2617677213",
   "gen9championsvgc2026regma-2617912311",
   "gen9championsvgc2026regma-2617667808",
   "gen9championsvgc2026regma-2621840106",
   "gen9championsvgc2026regma-2617823085",
   "gen9championsvgc2026regma-2617892221",
   "gen9championsvgc2026regma-2621838480",
   "gen9championsvgc2026regma-2621834518",
   "gen9championsvgc2026regma-2620060286",
   "gen9championsvgc2026regma-2619550401",
   "gen9championsvgc2026regma-2621843791",
   "gen9championsvgc2026regma-2619553764",
   "gen9championsvgc2026regma-2620044313",
   "gen9championsvgc2026regma-2617882249",
   "gen9championsvgc2026regma-2617789115",
   "gen9championsvgc2026regma-2617886017",
   "gen9championsvgc2026regma-2620054754",
   "gen9championsvgc2026regma-2619545581",
   "gen9championsvgc2026regma-2621842327",
   "gen9championsvgc2026regma-2621617372",
   "gen9championsvgc2026regma-2618053969",
   "gen9championsvgc2026regma-2618059713",
   "gen9championsvgc2026regma-2621836424",
   "gen9championsvgc2026regma-2631933654",
   "gen9championsvgc2026regma-2634777015",
   "gen9championsvgc2026regma-2631920730",
   "gen9championsvgc2026regma-2631932235",
   "gen9championsvgc2026regma-2632617779",
   "gen9championsvgc2026regma-2631926514",
   "gen9championsvgc2026regma-2630285523",
   "gen9championsvgc2026regma-2634073646",
   "gen9championsvgc2026regma-2625995445",
   "gen9championsvgc2026regma-2634077161",
   "gen9championsvgc2026regma-2631922880",
   "gen9championsvgc2026regma-2631924621",
   "gen9championsvgc2026regma-2632635514",
   "gen9championsvgc2026regma-2630852967",
   "gen9championsvgc2026regma-2631596101",
   "gen9championsvgc2026regma-2634071417",
   "gen9championsvgc2026regma-2634250150",
   "gen9championsvgc2026regma-2631928383",
   "gen9championsvgc2026regma-2632620439",
   "gen9championsvgc2026regma-2634252952",
   "gen9championsvgc2026regma-2633857142",
   "gen9championsvgc2026regma-2632629062",
   "gen9championsvgc2026regma-2632632557",
   "gen9championsvgc2026regma-2634781158",
   "gen9championsvgc2026regma-2625895063",
   "gen9championsvgc2026regma-2632614015",
   "gen9championsvgc2026regma-2630850280",
   "gen9championsvgc2026regma-2633855691"
  ]
 }
}
```