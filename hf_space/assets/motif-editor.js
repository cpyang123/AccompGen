() => {
  const mount = () => {
    const root = document.getElementById('motif-editor');
    if (!root || root.dataset.ready) return;
    root.dataset.ready = 'true';
    const $ = selector => root.querySelector(selector);
    const letters = 'CDEFGAB';
    const initial = () => [{p:28,a:'',d:'1'}, {p:32,a:'',d:'1/2'}, {p:31,a:'',d:'1/2'}, {p:30,a:'',d:'2'}];
    const state = {mode:'abstract', moves:[3,-1,-1], notes:initial(), bars:[], tool:'notes', selected:null, duration:'1', dotted:false, accidental:'', history:[]};
    let contourPositions=[0,3,2,1], contourDrag=null;
    const value = d => d.includes('/') ? Number(d.split('/')[0])/Number(d.split('/')[1]) : Number(d);
    const duration = () => String(value(state.duration) * (state.dotted ? 1.5 : 1));
    const pitch = n => letters[n.p % 7] + n.a + Math.floor(n.p / 7);
    const midi = n => 12 * (Math.floor(n.p / 7) + 1) + [0,2,4,5,7,9,11][n.p % 7] + (n.a === '#' ? 1 : n.a === 'b' ? -1 : 0);
    const snapshot = () => { state.history.push(JSON.stringify({notes:state.notes,bars:state.bars})); if (state.history.length > 40) state.history.shift(); };
    let audioContext, playing = [], drag = null;
    const stopAudio = () => { playing.forEach(node => { try { node.stop(); } catch (_) {} }); playing=[]; };
    const changed = () => { stopAudio(); renderNotes(); };
    window.motigenEditor = {getInput:() => ({mode:state.mode === 'abstract' ? 'Abstract motif' : 'Concrete notes', abstract:[0,...state.moves].join(','), notes:state.notes.flatMap((n,i) => state.bars.includes(i+1)?[[pitch(n),n.d],['|','']]:[[pitch(n),n.d]])})};
    const VF = window.VexFlow;
    const svgElement = (tag, attrs={}) => {
      const el=document.createElementNS('http://www.w3.org/2000/svg',tag);
      Object.entries(attrs).forEach(([key,val])=>el.setAttribute(key,val));return el;
    };
    const makeNote = (n, accidental=n.a) => {
      const beats=value(n.d), dotted=[.375,.75,1.5,3,6].includes(beats);
      const base=beats/(dotted?1.5:1);
      const note=new VF.StaveNote({keys:[letters[n.p%7].toLowerCase()+'/'+Math.floor(n.p/7)],
        duration:({'0.25':'16','0.5':'8','1':'q','2':'h','4':'w'})[base],autoStem:true});
      if(accidental)note.addModifier(new VF.Accidental(accidental));
      if(dotted)VF.Dot.buildAndAttach([note],{all:true});
      return note;
    };
    const placeNote = (context, stave, n, x, color, accidental=n.a) => {
      const note=makeNote(n,accidental);note.setStave(stave);
      VF.Formatter.SimpleFormat([note]);
      const tick=note.getTickContext(),center=(note.getNoteHeadBeginX()+note.getNoteHeadEndX())/2;
      tick.setX(tick.getX()+x-center);
      note.setStyle({fillStyle:color,strokeStyle:color});
      note.setContext(context).draw();return note;
    };
    const miniature = (n, color='#3d342a') => {
      const holder=document.createElement('div'),renderer=new VF.Renderer(holder,VF.Renderer.Backends.SVG);
      renderer.resize(40,60);
      const context=renderer.getContext();
      const stave=new VF.Stave(0,-41,40); // B4 sits at y=19; staff itself stays hidden.
      placeNote(context,stave,{...n,p:34},15,color);
      return holder.firstChild;
    };
    root.querySelectorAll('[data-duration]').forEach(button => {
      const svg=miniature({p:34,a:'',d:button.dataset.duration});
      svg.setAttribute('width','30');svg.setAttribute('height','30');
      svg.style.width='30px';svg.style.height='30px';svg.style.display='block';
      svg.setAttribute('viewBox','0 8 40 52');svg.setAttribute('aria-hidden','true');
      button.querySelector('span').replaceChildren(svg);
    });
    for(let p=21;p<=42;p++) $('#me-pitch').add(new Option(letters[p%7]+Math.floor(p/7),p));
    const contourGeometry = () => {
      const low=Math.min(-3,...contourPositions)-1, high=Math.max(4,...contourPositions)+1;
      return {low,high,step:180/(high-low)};
    };
    const syncContour = () => {state.moves=contourPositions.slice(1).map((p,i)=>Math.sign(p-contourPositions[i])*Math.min(3,Math.abs(p-contourPositions[i])));};
    function movePoint(index,position) {
      if(position===contourPositions[index-1]||position===contourPositions[index+1])return;
      contourPositions[index]=position;syncContour();renderContour();
    }
    function renderContour() {
      const {low,high,step}=contourDrag?.geometry || contourGeometry();
      const points=contourPositions.map((p,i)=>[25+i*430/(contourPositions.length-1),24+(high-p)*step]);
      const grid=Array.from({length:high-low+1},(_,i)=>`<path d="M15 ${24+i*step}H465" stroke="#e3d8c5" stroke-dasharray="3 5"/>`).join('');
      const labels=state.moves.map((m,i)=>`<text x="${(points[i][0]+points[i+1][0])/2}" y="265" text-anchor="middle" font-size="11" fill="#786346">${m>0?'↑':'↓'} ${['','Step','Skip','Leap'][Math.abs(m)]}</text>`).join('');
      $('#me-contour').innerHTML=grid+`<path d="${points.map(([x,y],i)=>`${i?'L':'M'}${x},${y}`).join(' ')}" fill="none" stroke="#987744" stroke-width="2.5"/>`+points.map(([x,y],i)=>`<g data-point="${i}" tabindex="0" role="slider" aria-label="Contour note ${i+1}" aria-orientation="vertical" aria-valuemin="-27" aria-valuemax="27" aria-valuenow="${contourPositions[i]}" aria-valuetext="${i===0?'Starting note':(state.moves[i-1]>0?'Up ':'Down ')+['','step','skip','leap'][Math.abs(state.moves[i-1])]}" transform="translate(${x} ${y})"><circle r="22" fill="transparent"/><circle class="me-point" r="8" fill="${contourDrag?.index===i?'#95652a':'#51412e'}" stroke="#fff" stroke-width="2"/></g><text x="${x}" y="239" text-anchor="middle" font-size="11" fill="#786952">${i+1}</text>`).join('')+labels;
    }
    function renderMoves(reset=false) {
      if(reset){contourPositions=[0];state.moves.forEach(move=>contourPositions.push(contourPositions.at(-1)+move));}
      $('#me-length').textContent=`${state.moves.length+1} notes`;
      $('#me-shorter').disabled=state.moves.length===3;$('#me-longer').disabled=state.moves.length===9;
      renderContour();
    }
    const focusNote = () => { if(state.selected!==null) $(`[data-note="${state.selected}"]`)?.focus({preventScroll:true}); };
    function selectNote(i) {
      state.selected=i;
      if(i!==null){
        const n=state.notes[i], beats=value(n.d);state.dotted=[.375,.75,1.5,3,6].includes(beats);
        const base=beats/(state.dotted?1.5:1);
        state.duration=({'0.25':'1/4','0.5':'1/2'})[base] || String(base);state.accidental=n.a;
      }
      renderNotes();focusNote();
    }
    function renderNotes() {
      const hadNoteFocus=Boolean(document.activeElement?.closest('#me-staff [data-note]'));
      if(state.selected!==null && !state.notes[state.selected]) state.selected=null;
      const selected=state.notes[state.selected];
      root.querySelectorAll('[data-duration]').forEach(b=>b.setAttribute('aria-pressed',String(state.tool==='notes'&&b.dataset.duration===state.duration)));
      $('#me-barline').setAttribute('aria-pressed',String(state.tool==='barline'));
      $('#me-dotted').checked=state.dotted;$('#me-accidental').value=state.accidental;
      $('#me-pitch').disabled=!selected;if(selected) $('#me-pitch').value=selected.p;
      $('#me-delete').disabled=!selected;$('#me-undo').disabled=!state.history.length;$('#me-listen').disabled=!state.notes.length;
      const holder=document.createElement('div'),renderer=new VF.Renderer(holder,VF.Renderer.Backends.SVG);
      renderer.resize(680,272);
      const context=renderer.getContext(),svg=holder.firstChild;
      svg.setAttribute('pointer-events','auto');svg.setAttribute('data-engraver','vexflow');
      const stave=new VF.Stave(18,16,642,{spacingBetweenLinesPx:16,leftBar:false,rightBar:false});
      stave.addClef('treble').setStyle({strokeStyle:'#b6a78e',fillStyle:'#66513a'}).setContext(context).draw();
      const accidentals=new Map();
      state.notes.forEach((n,i)=>{
        if(state.bars.includes(i))accidentals.clear();
        const x=110+i*52,color=i===state.selected?'#95652a':'#3d342a';
        const group=context.openGroup('editable-note');
        Object.entries({'data-note':i,role:'button',tabindex:0,'aria-label':`Note ${i+1}: ${pitch(n)}, ${value(n.d)} beats`}).forEach(([key,val])=>group.setAttribute(key,val));
        group.appendChild(svgElement('rect',{x:x-23,y:25,width:46,height:210,rx:7,stroke:'none',fill:i===state.selected?'#eee0c6':'transparent'}));
        // A natural is engraved only when cancelling an earlier accidental in this bar.
        const prior=accidentals.get(n.p)||'', accidental=n.a===prior?'':n.a||'n';
        placeNote(context,stave,n,x,color,accidental);accidentals.set(n.p,n.a);
        context.closeGroup();
      });
      state.bars.forEach(boundary=>{
        const x=110+(boundary-.5)*52,group=context.openGroup('editable-barline');
        Object.entries({'data-bar-after':boundary,role:'button',tabindex:0,'aria-label':`Remove barline after note ${boundary}`}).forEach(([key,val])=>group.setAttribute(key,val));
        group.appendChild(svgElement('rect',{x:x-9,y:70,width:18,height:84,stroke:'none',fill:'transparent'}));
        new VF.Barline(VF.BarlineType.SINGLE).setX(x).setStave(stave).setContext(context).draw();
        context.closeGroup();
      });
      $('#me-staff').replaceChildren(svg,svgElement('g',{id:'me-drop-preview','pointer-events':'none'}));
      const repeated=state.notes.some((n,i)=>i>0 && midi(n)===midi(state.notes[i-1]));
      const invalid=state.notes.length<4 || repeated;
      $('#me-note-status').dataset.invalid=String(invalid);
      $('#me-note-status').textContent=`${state.notes.length}/10 notes · `+(state.notes.length<4?`Add ${4-state.notes.length} more to generate.`:repeated?'Combine or change repeated pitches.':'Ready');
      $('#me-selection').textContent=state.tool==='barline'?'Tap a gap to add · tap a line to remove':selected?`Note ${state.selected+1} · ${pitch(selected)} · ${value(selected.d)} beats`:'';
      if(hadNoteFocus)focusNote();
    }
    const coords = (x,y) => {const svg=$('#me-staff'),r=svg.getBoundingClientRect(),clip=svg.parentElement.getBoundingClientRect();return {x:(x-r.left)*680/r.width,y:(y-r.top)*272/r.height,inside:x>=Math.max(r.left,clip.left)&&x<=Math.min(r.right,clip.right)&&y>=r.top&&y<=r.bottom};};
    const position = c => Math.max(21,Math.min(42,Math.round(30+(144-c.y)/8)));
    const barAt = c => Math.max(1,Math.min(state.notes.length,Math.round((c.x-110)/52+.5)));
    function addBar(boundary) {
      if(!state.notes.length || state.bars.includes(boundary))return;
      snapshot();state.bars.push(boundary);state.bars.sort((a,b)=>a-b);changed();
    }
    function removeBar(boundary) {
      snapshot();state.bars=state.bars.filter(b=>b!==boundary);changed();
    }
    function deleteNote(index) {
      state.notes.splice(index,1);
      state.bars=[...new Set(state.bars.map(b=>b>index?b-1:b))].filter(b=>b>0&&b<=state.notes.length);
    }
    function insert(c,d) {
      if(state.notes.length>=10){$('#me-note-status').textContent='Maximum 10 notes. Delete a note before adding another.';return;}
      snapshot();const index=Math.max(0,Math.min(state.notes.length,Math.round((c.x-110)/52)));
      state.bars=state.bars.map(b=>b>index?b+1:b);
      state.notes.splice(index,0,{p:position(c),a:state.accidental,d});state.selected=index;changed();focusNote();
    }
    function setDuration(d) {
      state.tool='notes';state.duration=d;
      if(state.selected!==null){snapshot();state.notes[state.selected].d=duration();}
      changed();
    }
    root.addEventListener('change',event=>{
      const el=event.target;
      if(el.id==='me-dotted'){state.dotted=el.checked;if(state.selected!==null){snapshot();state.notes[state.selected].d=duration();}changed();}
      if(el.id==='me-accidental'){state.accidental=el.value;if(state.selected!==null){snapshot();state.notes[state.selected].a=el.value;}changed();}
      if(el.id==='me-pitch'&&state.selected!==null){snapshot();state.notes[state.selected].p=Number(el.value);changed();}
    });
    const setMode = button => {state.mode=button.dataset.mode;root.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-selected',String(b===button)));$('#me-abstract').hidden=state.mode!=='abstract';$('#me-concrete').hidden=state.mode!=='concrete';stopAudio();};
    let modeTap=null,suppressClick=false;
    root.addEventListener('click',event=>{
      const button=event.target.closest('button');if(!button) return;
      if(button.dataset.mode)setMode(button);
      if(button.id==='me-longer'&&state.moves.length<9){contourPositions.push(contourPositions.at(-1)+(contourPositions.at(-1)<27?1:-1));syncContour();renderMoves();}
      if(button.id==='me-shorter'&&state.moves.length>3){contourPositions.pop();syncContour();renderMoves();}
      if(button.dataset.shape){state.moves=button.dataset.shape==='arch'?[3,-1,-1]:button.dataset.shape==='rise'?[1,1,2,1,1]:[2,-1,3,-2,-1];renderMoves(true);}
      if(button.id==='me-barline'){if(!suppressClick){state.tool=state.tool==='barline'?'notes':'barline';state.selected=null;renderNotes();}suppressClick=false;}
      if(button.dataset.duration){if(!suppressClick)setDuration(button.dataset.duration);suppressClick=false;}
      if(button.id==='me-staff-left'||button.id==='me-staff-right')$('.me-staff-scroll').scrollBy({left:button.id==='me-staff-left'?-220:220,behavior:'smooth'});
      if(button.id==='me-undo'&&state.history.length){const previous=JSON.parse(state.history.pop());state.notes=previous.notes;state.bars=previous.bars;state.selected=null;changed();}
      if(button.id==='me-delete'&&state.selected!==null){snapshot();deleteNote(state.selected);state.selected=null;changed();}
      if(button.id==='me-clear'){snapshot();state.notes=[];state.bars=[];state.selected=null;changed();}
      if(button.id==='me-reset'){snapshot();state.notes=initial();state.bars=[];state.tool='notes';state.selected=null;changed();}
      if(button.id==='me-listen'){
        stopAudio();audioContext ||= new (window.AudioContext || window.webkitAudioContext)();audioContext.resume();
        let start=audioContext.currentTime+.05;
        for(const n of state.notes){const length=value(n.d)*.6,osc=audioContext.createOscillator(),gain=audioContext.createGain();osc.type='triangle';osc.frequency.value=440*Math.pow(2,(midi(n)-69)/12);gain.gain.setValueAtTime(0,start);gain.gain.linearRampToValueAtTime(.17,start+.01);gain.gain.exponentialRampToValueAtTime(.001,start+Math.max(.02,length-.01));osc.connect(gain);gain.connect(audioContext.destination);osc.start(start);osc.stop(start+length);playing.push(osc);start+=length;}
      }
    });
    root.addEventListener('pointerdown',event=>{
      if(event.button!==0)return;
      const tab=event.target.closest('[data-mode]');
      if(tab){modeTap={button:tab,x:event.clientX,y:event.clientY};return;}
      const point=event.target.closest('[data-point]');
      if(point){
        event.preventDefault();
        contourDrag={index:Number(point.dataset.point),geometry:contourGeometry(),original:[...contourPositions],pointerId:event.pointerId};
        $('#me-contour').setPointerCapture(event.pointerId);renderContour();return;
      }
      const barTool=event.target.closest('[data-bar-tool]'),bar=event.target.closest('[data-bar-after]');
      if(bar){event.preventDefault();removeBar(Number(bar.dataset.barAfter));return;}
      const staffTarget=event.target.closest('#me-staff');
      if(barTool || (staffTarget&&state.tool==='barline')){
        drag={type:barTool?'barPalette':'barPlace',x:event.clientX,y:event.clientY,moved:false};
        if(staffTarget)event.preventDefault();return;
      }
      const palette=event.target.closest('[data-duration]'),note=event.target.closest('[data-note]'),staff=event.target.closest('#me-staff');
      if(!palette&&!staff)return;
      if(palette)state.tool='notes';
      if(note)selectNote(Number(note.dataset.note));
      drag={type:palette?'palette':note?'note':'empty',index:note?Number(note.dataset.note):null,d:palette?palette.dataset.duration:duration(),x:event.clientX,y:event.clientY,moved:false};
      if(staff)event.preventDefault();
    });
    window.addEventListener('pointermove',event=>{
      if(contourDrag){
        if(event.pointerId!==contourDrag.pointerId)return;
        const r=$('#me-contour').getBoundingClientRect(), y=(event.clientY-r.top)*280/r.height;
        const {low,high,step}=contourDrag.geometry;
        movePoint(contourDrag.index,Math.max(-27,Math.min(27,Math.max(low,Math.min(high,Math.round(high-(y-24)/step))))));return;
      }
      if(!drag)return;
      if(Math.hypot(event.clientX-drag.x,event.clientY-drag.y)>5)drag.moved=true;
      if(!drag.moved)return;
      const c=coords(event.clientX,event.clientY),preview=$('#me-drop-preview');
      if(drag.type==='barPalette'||drag.type==='barPlace'){
        const x=110+(barAt(c)-.5)*52;
        if(preview)preview.innerHTML=c.inside&&state.notes.length?`<path d="M${x} 80V144" stroke="#95652a" stroke-width="3" opacity=".6"/>`:'';return;
      }
      if(preview){
        preview.replaceChildren();
        if(c.inside){
          const holder=document.createElement('div'),renderer=new VF.Renderer(holder,VF.Renderer.Backends.SVG);
          renderer.resize(680,272);holder.firstChild.setAttribute('opacity','.5');
          const d=drag.type==='note'?state.notes[drag.index].d:String(value(drag.d)*(drag.type==='palette'&&state.dotted?1.5:1));
          placeNote(renderer.getContext(),new VF.Stave(18,16,642,{spacingBetweenLinesPx:16}),
            {p:position(c),a:state.accidental,d},Math.max(100,Math.min(640,c.x)),'#95652a');
          preview.appendChild(holder.firstChild);
        }
      }
    });
    window.addEventListener('pointerup',event=>{
      if(modeTap){const tap=modeTap;modeTap=null;if(event.target.closest('[data-mode]')===tap.button&&Math.hypot(event.clientX-tap.x,event.clientY-tap.y)<8)setMode(tap.button);return;}
      if(contourDrag){
        if(event.pointerId!==contourDrag.pointerId)return;
        const index=contourDrag.index;contourDrag=null;renderContour();$(`[data-point="${index}"]`).focus({preventScroll:true});return;
      }
      if(!drag)return;const action=drag;drag=null;
      const c=coords(event.clientX,event.clientY);$('#me-drop-preview').innerHTML='';
      if(action.type==='barPalette'||action.type==='barPlace'){
        if(c.inside&&(action.moved||action.type==='barPlace'))addBar(barAt(c));
        if(action.moved){suppressClick=true;setTimeout(()=>suppressClick=false,0);}return;
      }
      if(action.type==='palette'){
        if(action.moved){suppressClick=true;if(c.inside){state.duration=action.d;insert(c,String(value(action.d)*(state.dotted?1.5:1)));}setTimeout(()=>suppressClick=false,0);}
      }else if(action.type==='note'){
        if(action.moved&&c.inside){snapshot();const n=state.notes.splice(action.index,1)[0];n.p=position(c);const index=Math.max(0,Math.min(state.notes.length,Math.round((c.x-110)/52)));state.notes.splice(index,0,n);state.selected=index;changed();focusNote();}
      }else if(c.inside&&!action.moved)insert(c,duration());
    });
    window.addEventListener('pointercancel',()=>{modeTap=null;if(contourDrag){contourPositions=contourDrag.original;contourDrag=null;syncContour();renderContour();}drag=null;$('#me-drop-preview').innerHTML='';});
    root.addEventListener('keydown',event=>{
      const point=event.target.closest('[data-point]');
      if(point){
        if(!['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(event.key))return;
        event.preventDefault();let index=Number(point.dataset.point);
        if(event.key==='ArrowUp'||event.key==='ArrowDown'){
          const direction=event.key==='ArrowUp'?1:-1;let position=contourPositions[index]+direction;
          while(position===contourPositions[index-1]||position===contourPositions[index+1])position+=direction;
          if(position>=-27&&position<=27)movePoint(index,position);
        }else index=Math.max(0,Math.min(contourPositions.length-1,index+(event.key==='ArrowRight'?1:-1)));
        $(`[data-point="${index}"]`).focus({preventScroll:true});return;
      }
      const bar=event.target.closest('[data-bar-after]');
      if(bar&&['Delete','Backspace','Enter',' '].includes(event.key)){
        event.preventDefault();const boundary=Number(bar.dataset.barAfter);removeBar(boundary);
        $(`[data-note="${boundary-1}"]`)?.focus({preventScroll:true});return;
      }
      if(event.key==='Escape'&&state.tool==='barline'){state.tool='notes';renderNotes();return;}
      const target=event.target.closest('[data-note]');if(!target)return;
      const i=Number(target.dataset.note);
      if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Delete','Backspace','Enter',' '].includes(event.key))event.preventDefault();else return;
      if(event.key==='ArrowUp'||event.key==='ArrowDown'){snapshot();state.notes[i].p=Math.max(21,Math.min(42,state.notes[i].p+(event.key==='ArrowUp'?1:-1)));state.selected=i;changed();}
      else if(event.key==='Delete'||event.key==='Backspace'){snapshot();deleteNote(i);state.selected=Math.min(i,state.notes.length-1);if(state.selected<0)state.selected=null;changed();}
      else selectNote(Math.max(0,Math.min(state.notes.length-1,i+(event.key==='ArrowRight'?1:event.key==='ArrowLeft'?-1:0))));
      if(state.selected!==null)$(`[data-note="${state.selected}"]`).focus();
    });
    renderMoves();renderNotes();
    if (!window.motigenCatTimer) {
  const CAT_INTERVAL_MS = 180000; // a cat every three minutes
  const CAT_SPEED = 55;           // px per second
  const CAT_SVG =
    '<svg viewBox="0 0 120 64" xmlns="http://www.w3.org/2000/svg">' +
    '<g fill="currentColor">' +
    '<path class="tail" d="M18 34 C6 30 2 18 9 9" fill="none" stroke="currentColor" stroke-width="5" stroke-linecap="round"/>' +
    '<rect class="leg" x="22" y="36" width="5.5" height="24" rx="2.7"/>' +
    '<rect class="leg off" x="32" y="36" width="5.5" height="24" rx="2.7"/>' +
    '<rect class="leg off" x="60" y="36" width="5.5" height="24" rx="2.7"/>' +
    '<rect class="leg" x="70" y="36" width="5.5" height="24" rx="2.7"/>' +
    '<ellipse cx="49" cy="34" rx="32" ry="13"/>' +
    '<circle cx="88" cy="22" r="11"/>' +
    '<path d="M79 15 L77 3 L87 11 Z"/>' +
    '<path d="M90 13 L97 3 L98 16 Z"/>' +
    '</g></svg>';
  let goRight = true;
  function walkCat() {
    if (document.hidden) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (document.querySelector('.cat-walker')) return;
    const cat = document.createElement('div');
    cat.className = 'cat-walker ' + (goRight ? 'ltr' : 'rtl');
    cat.setAttribute('aria-hidden', 'true');
    cat.style.setProperty('--cat-dur', Math.round((window.innerWidth + 140) / CAT_SPEED) + 's');
    cat.innerHTML = '<div class="cat-bob">' + CAT_SVG + '</div>';
    let lane=document.getElementById('motigen-cat-lane');
    if(!lane){lane=document.createElement('div');lane.id='motigen-cat-lane';lane.setAttribute('aria-hidden','true');document.body.appendChild(lane);}
    lane.style.visibility='hidden';lane.appendChild(cat);
    // An embedded Space may be taller than the browser screen. An implicit-root
    // observer clips through ancestor frames and reports the visible rectangle
    // in this document's viewport coordinates, without reading the parent DOM.
    const probe=document.createElement('div');
    probe.style.cssText='position:fixed;inset:0;pointer-events:none;visibility:hidden';
    probe.setAttribute('aria-hidden','true');document.body.appendChild(probe);
    let refreshTimer,expiryTimer;
    const cleanup=()=>{clearTimeout(refreshTimer);clearTimeout(expiryTimer);observer.disconnect();probe.remove();cat.remove();};
    const observer=new IntersectionObserver(([entry])=>{
      if(!cat.isConnected){cleanup();return;}
      const r=entry.intersectionRect,v=window.visualViewport;
      const bottom=Math.min(r.bottom,v?v.offsetTop+v.height:innerHeight);
      lane.style.top=Math.max(r.top,bottom-40)+'px';lane.style.bottom='auto';
      lane.style.left=r.left+'px';lane.style.right='auto';lane.style.width=r.width+'px';
      lane.style.setProperty('--cat-width',r.width+'px');
      lane.style.visibility=r.width>0&&bottom-r.top>=40?'visible':'hidden';
      // Reobserve while walking: parent scrolling can move the visible rectangle
      // without changing its area, so intersection thresholds alone miss it.
      refreshTimer=setTimeout(()=>{if(!cat.isConnected){cleanup();return;}observer.unobserve(probe);observer.observe(probe);},100);
    });
    observer.observe(probe);
    cat.addEventListener('animationend',e=>{if(e.target===cat)cleanup();});
    expiryTimer=setTimeout(cleanup,(Math.round((window.innerWidth+140)/CAT_SPEED)+1)*1000);
    goRight = !goRight;
  }
  window.motigenCatTimer = setInterval(walkCat, CAT_INTERVAL_MS);
  window.walkCat = walkCat; // run walkCat() in the console to summon one now

    }
  };
  mount();
  const observer = new MutationObserver(mount);
  observer.observe(document.body,{childList:true,subtree:true});
}
