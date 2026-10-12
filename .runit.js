
function mkEl(){ return new Proxy({dataset:{},style:{},clientWidth:500,
  set innerHTML(v){}, get innerHTML(){return "";},
  set textContent(v){}, appendChild(c){return c;}, after(){}, before(){},
  addEventListener(){}, removeEventListener(){}, setAttribute(){}, getAttribute(){return null;},
  querySelector(){return mkEl();}, querySelectorAll(){return [];},
  classList:{add(){},remove(){},toggle(){},contains(){return false;}},
  click(){}, remove(){}}, { get(t,k){ return k in t ? t[k] : (()=>mkEl()); } }); }
const document = { getElementById(){return mkEl();}, querySelector(){return mkEl();},
  querySelectorAll(){return [];}, createElement(){return mkEl();},
  createElementNS(){return mkEl();}, addEventListener(){}, fullscreenElement:null, exitFullscreen(){} };
const window = { addEventListener(){}, lucide:null };
