// Valida equipos packed (uno por linea en stdin) con el TeamValidator del Showdown fijado.
// Uso: node scripts/validate_teams.js <ruta a pokemon-showdown> < teams.txt
// Salida: una linea por equipo, "OK" o la lista de problemas en JSON.
const path = require('path');
const sd = path.join(process.argv[2], 'dist', 'sim');
const {TeamValidator, Teams} = require(sd);
const v = TeamValidator.get(process.argv[3] || 'gen9championsvgc2026regma');
const lines = require('fs').readFileSync(0, 'utf8').split('\n');
const out = [];
for (const l of lines) {
  if (!l.trim()) { out.push('EMPTY'); continue; }
  const probs = v.validateTeam(Teams.unpack(l));
  out.push(probs ? JSON.stringify(probs) : 'OK');
}
process.stdout.write(out.join('\n'));
