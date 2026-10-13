'use strict';
let data;
let selectedSample = 0;
let selectedMetric = 'p50_request_ms';
const $ = id => document.getElementById(id);
const median = values => { const v = [...values].sort((a,b)=>a-b); const m = Math.floor(v.length/2); return v.length%2 ? v[m] : (v[m-1]+v[m])/2; };
const format = value => Math.round(value).toLocaleString('en-US');
function updateMetric(metric) {
  selectedMetric = metric;
  if (!data) return;
  const a = data.summary.baseline[metric], b = data.summary.optimized[metric];
  $('metric-baseline').textContent = format(a)+' ms';
  $('metric-optimized').textContent = format(b)+' ms';
  $('aggregate-baseline').style.width = (a/Math.max(a,b)*100)+'%';
  $('aggregate-optimized').style.width = (b/Math.max(a,b)*100)+'%';
  $('metric-speedup').textContent = (a/b).toFixed(2)+'× speedup';
  $('metric-reduction').textContent = ((1-b/a)*100).toFixed(1)+'% less latency';
  $('metric-note').textContent = metric === 'p50_decoder_ms' ? 'Decoder-only timing; excludes both encoders and feature/text processing. Percentiles are computed across per-utterance repetition medians.' : 'Includes feature processing, encoders, decoding and text processing. Excludes model loading and file I/O. Percentiles are computed across per-utterance repetition medians.';
  document.querySelectorAll('[data-metric]').forEach(button => { const active = button.dataset.metric===metric; button.classList.toggle('active',active); button.setAttribute('aria-pressed',String(active)); });
}
function renderChart() {
  const root = $('sample-chart');
  root.replaceChildren();
  const maximum = Math.max(...data.samples.flatMap(s=>['baseline','optimized'].map(m=>median(s.runs[m].map(r=>r.request_ms)))));
  data.samples.forEach((s,index)=>{
    const button = document.createElement('button');
    button.className = 'clip-bar'; button.dataset.index = index;
    const a = median(s.runs.baseline.map(r=>r.request_ms)), b = median(s.runs.optimized.map(r=>r.request_ms));
    button.title = s.id+': target '+format(a)+' ms, draft + verify '+format(b)+' ms';
    button.setAttribute('aria-label','Select utterance '+s.id+'. '+button.title);
    ['baseline','optimized'].forEach((mode,i)=>{const bar=document.createElement('span');bar.className=mode;bar.style.height=([a,b][i]/maximum*95)+'%';button.append(bar);});
    button.addEventListener('click',()=>selectSample(index));root.append(button);
  });
}
function selectSample(index) {
  selectedSample = index;
  const s = data.samples[index], base = s.runs.baseline, opt = s.runs.optimized;
  const a=median(base.map(r=>r.request_ms)),b=median(opt.map(r=>r.request_ms));
  $('sample-select').value = String(index);
  $('sample-duration').textContent = s.audio_seconds.toFixed(2)+' seconds of audio';
  $('sample-speedup').textContent = (a/b).toFixed(2)+'× median request speedup';
  const hashesMatch=base.every((r,i)=>r.token_sha256===opt[i].token_sha256);
  $('sample-agreement').textContent = s.exact_token_match&&hashesMatch?'✓ Exact token agreement':'Token agreement failed';
  $('transcript-baseline').textContent = s.predictions.baseline;
  $('transcript-optimized').textContent = s.predictions.optimized;
  $('sample-runs').replaceChildren();
  base.forEach((r,i)=>{
    const tr = document.createElement('tr');
    const fields=[String(i+1),format(r.request_ms)+' ms',format(opt[i].request_ms)+' ms',r.target_calls+' → '+opt[i].target_calls,r.token_sha256===opt[i].token_sha256?'SHA256: matched':'Mismatch'];
    fields.forEach(value=>{const cell=document.createElement('td');cell.textContent=value;tr.append(cell);});$('sample-runs').append(tr);
  });
  document.querySelectorAll('.clip-bar').forEach((button,i)=>{button.classList.toggle('selected',i===index);button.setAttribute('aria-pressed',String(i===index));});
}
document.querySelectorAll('[data-metric]').forEach(button=>button.addEventListener('click',()=>updateMetric(button.dataset.metric)));
$('sample-select').addEventListener('change',event=>selectSample(Number(event.target.value)));
fetch('data.json').then(response=>{if(!response.ok)throw new Error('Data unavailable');return response.json();}).then(value=>{
  data=value;
  const selector=$('sample-select');selector.replaceChildren();
  data.samples.forEach((s,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=String(i+1).padStart(2,'0')+' · '+s.id;selector.append(option);});
  renderChart();updateMetric(selectedMetric);selectSample(selectedSample);
}).catch(()=>{$('load-error').hidden=false;});
const steps = [
  {label:'DRAFT',heading:'Tiny drafts four tokens.',description:'A small draft model proposes a short continuation from the current prefix. It can run quickly, but its guesses have no authority over the final output.',draft:[['we',''],['are',''],['happy',''],['to','']],target:[['Waiting for verification','placeholder']],committed:[['… existing prefix','placeholder']],cache:'Both models condition on the committed prefix.'},
  {label:'VERIFY',heading:'Large checks the whole block.',description:'The target model verifies the proposed block in one pass. The first two tokens match its greedy predictions. At the third position, the target prefers “glad” rather than “happy”.',draft:[['we','accept'],['are','accept'],['happy','mismatch'],['to','discard']],target:[['we','accept'],['are','accept'],['glad','correction'],['…','placeholder']],committed:[['… existing prefix','placeholder']],cache:'Only the contiguous matching prefix may be accepted.'},
  {label:'COMMIT',heading:'Accept the prefix and correction.',description:'The decoder commits “we are” from the draft and then the target’s correction, “glad”. Later draft tokens remain uncommitted because their context contained the rejected token.',draft:[['we','accept'],['are','accept'],['happy','discard'],['to','discard']],target:[['we','accept'],['are','accept'],['glad','correction']],committed:[['…','placeholder'],['we','accept'],['are','accept'],['glad','correction']],cache:'Committed continuation: we are glad.'},
  {label:'ROLL BACK',heading:'Repair the cache. Continue.',description:'Discard the speculative self-attention cache suffix. Preserve the encoder cross-attention state, align draft and target with the committed prefix, and start the next block.',draft:[['Next block starts here','placeholder']],target:[['Verified prefix retained','placeholder']],committed:[['…','placeholder'],['we','accept'],['are','accept'],['glad','correction']],cache:'Rejected suffix removed; encoder cross-attention state reused.'}
];
let stepIndex=0;
function paintTokens(id,tokens){const root=$(id);root.replaceChildren();tokens.forEach(([text,state])=>{const token=document.createElement('span');token.className='token '+state;token.textContent=text;root.append(token);});}
function showStep(index){stepIndex=index;const s=steps[index];$('step-count').textContent='STEP '+(index+1)+' / 4';$('step-status').textContent=s.label;$('step-heading').textContent=s.heading;$('step-description').textContent=s.description;$('cache-note').textContent=s.cache;paintTokens('draft-tokens',s.draft);paintTokens('target-tokens',s.target);paintTokens('committed-tokens',s.committed);document.querySelectorAll('.step-list li').forEach((element,i)=>element.classList.toggle('current',i===index));$('walk-next').textContent=index===3?'Start again ↺':'Next step →';}
$('walk-next').addEventListener('click',()=>showStep((stepIndex+1)%steps.length));
$('walk-reset').addEventListener('click',()=>showStep(0));
showStep(0);
