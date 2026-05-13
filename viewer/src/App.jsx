import React, { useState, useEffect, useRef } from 'react';
import { 
  FileText, CheckCircle2, AlertCircle, ChevronRight, 
  Hash, ChevronLeft, List, ChevronDown, ChevronUp,
  Maximize2, Minimize2
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [data, setData] = useState({ documents: [], results: {} });
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [docContent, setDocContent] = useState('');
  const [activeCitations, setActiveCitations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedSections, setExpandedSections] = useState({});
  const docRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/data`)
      .then(res => {
        if (!res.ok) throw new Error(`Server responded with ${res.status}`);
        return res.json();
      })
      .then(d => {
        setData(d);
        if (d.documents && d.documents.length > 0) {
          setSelectedDoc(d.documents[0]);
        }
        setLoading(false);
      })
      .catch(err => {
        setError(`Failed to load data: ${err.message}. Ensure the Data Server is running on port 8000.`);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (selectedDoc) {
      fetch(`${API_BASE}/api/file/${selectedDoc}`)
        .then(res => res.text())
        .then(text => setDocContent(text));
    }
  }, [selectedDoc]);

  const getGlobalPageRange = (chunkIdx) => {
    const CHUNK_SIZE = 5;
    const STEP = 4;
    const startGlobalIdx = chunkIdx * STEP;
    const endGlobalIdx = startGlobalIdx + CHUNK_SIZE - 1;
    return { start: startGlobalIdx, end: endGlobalIdx };
  };

  const handleJump = (chunkIdx) => {
    const range = getGlobalPageRange(chunkIdx);
    setActiveCitations([chunkIdx]);
    
    setTimeout(() => {
      const el = document.getElementById(`global-page-${range.start}`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 50);
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const toggleAll = (expand) => {
    const newState = {};
    if (expand) {
      Object.keys(data.results).forEach(k => {
        if (k !== 'citations') newState[k] = true;
      });
    }
    setExpandedSections(newState);
  };

  const renderContent = () => {
    if (!docContent) return <div className="p-12 text-slate-400 font-medium">Select a document to begin...</div>;

    const pageMarkerRegex = /=== Page (\d+) ===/g;
    const parts = docContent.split(pageMarkerRegex);
    
    const elements = [];
    if (parts[0].trim()) {
        elements.push(<div key="header" className="p-10 border-b border-slate-100 text-slate-400 italic text-sm">{parts[0]}</div>);
    }

    let globalPageIdx = 0;
    for (let i = 1; i < parts.length; i += 2) {
      const pageLabel = parts[i];
      const pageText = parts[i + 1];
      const currentIdx = globalPageIdx;

      const isHighlighted = activeCitations.some(idx => {
        const range = getGlobalPageRange(idx);
        return currentIdx >= range.start && currentIdx <= range.end;
      });

      elements.push(
        <div 
          key={globalPageIdx} 
          id={`global-page-${globalPageIdx}`}
          className={`relative mb-10 p-10 transition-all rounded-3xl border ${
            isHighlighted 
              ? 'bg-blue-50 border-blue-300 ring-8 ring-blue-500/5 shadow-2xl z-10 scale-[1.01]' 
              : 'bg-white border-slate-100 shadow-sm opacity-90'
          }`}
        >
          <div className="absolute -top-4 left-8 flex gap-2">
            <span className="px-3 py-1 bg-slate-900 text-white text-[10px] font-black rounded-full uppercase tracking-widest shadow-md">
              Page {pageLabel}
            </span>
            <span className="px-3 py-1 bg-slate-200 text-slate-500 text-[10px] font-black rounded-full uppercase tracking-widest">
              Index {globalPageIdx}
            </span>
          </div>
          <pre className="whitespace-pre-wrap font-mono text-[13px] leading-relaxed text-slate-700">
            {pageText.trim()}
          </pre>
        </div>
      );
      globalPageIdx++;
    }

    return <div className="p-12 space-y-4">{elements}</div>;
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-[#f8f9fa] text-slate-800">
        <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-6"></div>
        <p className="text-lg font-black animate-pulse tracking-widest uppercase">Initializing UI...</p>
      </div>
    );
  }

  const allExpanded = Object.keys(data.results).filter(k => k !== 'citations').every(k => expandedSections[k]);

  return (
    <div className="flex h-screen bg-white overflow-hidden text-slate-800 font-sans">
      {/* Left Sidebar (Only one meant to feel separate) */}
      <div className="w-72 bg-slate-50 border-r border-slate-200 flex flex-col z-10 shadow-sm">
        <div className="p-6 border-b border-slate-200 bg-white">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-lg shadow-lg shadow-blue-600/20 text-white">
              <FileText size={18} />
            </div>
            <div>
              <h1 className="text-sm font-black text-slate-900 tracking-tighter leading-none uppercase">Bidpanion</h1>
              <p className="text-[9px] text-slate-400 mt-1 uppercase font-black tracking-widest">Global Verifier</p>
            </div>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {data.documents.map(doc => (
            <button
              key={doc}
              onClick={() => setSelectedDoc(doc)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all text-left ${
                selectedDoc === doc 
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20' 
                  : 'hover:bg-slate-200/50 text-slate-500'
              }`}
            >
              <FileText size={16} />
              <span className="text-[10px] font-black truncate uppercase tracking-wider">{doc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Parallel Workspace Container */}
      <div className="flex-1 flex overflow-hidden">
        {/* Document View Panel */}
        <div className="flex-1 overflow-y-auto bg-slate-50/30 scroll-smooth border-r border-slate-100" ref={docRef}>
          <div className="sticky top-0 z-20 bg-white/90 backdrop-blur-md px-10 py-5 border-b border-slate-200 flex justify-between items-center">
            <div className="flex items-center gap-3">
               <span className="px-2 py-1 bg-slate-100 text-[9px] font-black text-slate-500 rounded uppercase tracking-widest">Document</span>
               <ChevronRight size={14} className="text-slate-300" />
               <span className="text-xs font-bold text-slate-900">{selectedDoc}</span>
            </div>
          </div>
          <div className="max-w-4xl mx-auto">
            {renderContent()}
          </div>
        </div>

        {/* Results Analysis Panel (No shadow, strictly parallel) */}
        <div className="w-[500px] flex flex-col bg-white overflow-hidden">
          <div className="px-8 py-5 border-b border-slate-200 flex justify-between items-center bg-white">
            <h2 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-900 flex items-center gap-2">
              <List size={14} className="text-blue-600" /> Extraction Results
            </h2>
            <button 
              onClick={() => toggleAll(!allExpanded)}
              className="flex items-center gap-2 px-3 py-1.5 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg transition-all text-[9px] font-black uppercase tracking-widest text-slate-600"
            >
              {allExpanded ? <Minimize2 size={10} /> : <Maximize2 size={10} />}
            </button>
          </div>
          
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {Object.entries(data.results).map(([section, value]) => {
              if (section === 'citations') return null;
              const isExpanded = expandedSections[section];
              const isObject = typeof value === 'object' && value !== null && !Array.isArray(value);

              return (
                <div key={section} className={`border rounded-2xl overflow-hidden transition-all ${isExpanded ? 'border-blue-100 bg-blue-50/5' : 'border-slate-100 bg-white'}`}>
                  <button 
                    onClick={() => toggleSection(section)}
                    className={`w-full flex items-center justify-between px-5 py-4 transition-colors ${isExpanded ? 'bg-blue-50/30' : 'hover:bg-slate-50/50'}`}
                  >
                    <h3 className={`text-[10px] font-black uppercase tracking-[0.1em] ${isExpanded ? 'text-blue-600' : 'text-slate-500'}`}>{section}</h3>
                    {isExpanded ? <ChevronUp size={14} className="text-blue-500" /> : <ChevronDown size={14} className="text-slate-400" />}
                  </button>
                  
                  {isExpanded && (
                    <div className="p-5 space-y-4 bg-white/50 border-t border-blue-50/50">
                      {isObject ? (
                         Object.entries(value).map(([field, val]) => (
                           <FieldCard key={field} title={field} value={val} citationStr={data.results.citations?.sources[`${field}__quelle`]} onJump={handleJump} getRange={getGlobalPageRange} />
                         ))
                      ) : (
                        <FieldCard title={section} value={value} citationStr={data.results.citations?.sources[`${section}__quelle`]} onJump={handleJump} getRange={getGlobalPageRange} />
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

function FieldCard({ title, value, citationStr, onJump, getRange }) {
  const [activeIndex, setActiveIndex] = useState(0);
  const citations = citationStr 
    ? String(citationStr).split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n))
    : [];

  const handleNext = (e) => {
    e.stopPropagation();
    const next = (activeIndex + 1) % citations.length;
    setActiveIndex(next);
    onJump(citations[next]);
  };

  const handlePrev = (e) => {
    e.stopPropagation();
    const prev = (activeIndex - 1 + citations.length) % citations.length;
    setActiveIndex(prev);
    onJump(citations[prev]);
  };

  const isFound = value && value !== "Not found" && (Array.isArray(value) ? value.length > 0 : true);
  const currentRange = citations.length > 0 ? getRange(citations[activeIndex]) : null;
  
  return (
    <div 
      className={`group p-5 rounded-xl transition-all border ${
        citations.length > 0 ? 'hover:border-blue-500/50 bg-white cursor-pointer border-slate-100 shadow-sm' : 'bg-slate-50/50 border-slate-100 opacity-60'
      }`}
      onClick={() => citations.length > 0 && onJump(citations[activeIndex])}
    >
      <div className="flex justify-between items-start mb-3">
        <h4 className="text-[9px] font-black text-slate-400 uppercase tracking-widest leading-tight pr-4">{title}</h4>
        {citations.length > 0 && (
          <div className="flex items-center gap-1 bg-slate-50 p-1 rounded-lg border border-slate-100">
             <button onClick={handlePrev} className="p-1 hover:bg-white rounded-md transition-colors text-slate-400"><ChevronLeft size={12} /></button>
             <div className="px-2 text-[8px] font-black text-slate-600 min-w-[30px] text-center">
               {activeIndex + 1}/{citations.length}
             </div>
             <button onClick={handleNext} className="p-1 hover:bg-white rounded-md transition-colors text-slate-400"><ChevronRight size={12} /></button>
          </div>
        )}
      </div>

      <div className="text-[12px] text-slate-700 leading-relaxed font-medium">
        {Array.isArray(value) ? (
          <ul className="space-y-1.5">
            {value.map((item, i) => (
              <li key={i} className="flex gap-2 items-start">
                <div className="w-1 h-1 rounded-full bg-blue-500 mt-2 shrink-0"></div>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        ) : typeof value === 'object' && value !== null ? (
          <div className="space-y-2">
            {Object.entries(value).map(([k, v]) => (
              <div key={k} className="bg-slate-50/50 p-2.5 rounded-lg border border-slate-100">
                <span className="text-[8px] text-slate-400 font-black uppercase block mb-0.5 tracking-widest">{k}</span>
                <span className="text-slate-800">{typeof v === 'object' ? JSON.stringify(v) : (v || 'Not found')}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="whitespace-pre-wrap">{value || 'No information found'}</p>
        )}
      </div>

      <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isFound ? (
            <div className="flex items-center gap-1 px-2 py-0.5 bg-emerald-50 text-emerald-600 rounded-full border border-emerald-100 text-[8px] font-black uppercase tracking-widest">
              <CheckCircle2 size={10} /> Found
            </div>
          ) : (
            <div className="flex items-center gap-1 px-2 py-0.5 bg-amber-50 text-amber-600 rounded-full border border-amber-100 text-[8px] font-black uppercase tracking-widest">
              <AlertCircle size={10} /> Missing
            </div>
          )}
        </div>
        
        {currentRange && (
          <div className="flex items-center gap-1 px-2 py-0.5 bg-blue-50 text-blue-600 rounded-full border border-blue-100 text-[8px] font-black uppercase tracking-widest">
             Idx {currentRange.start}-{currentRange.end}
          </div>
        )}
      </div>
    </div>
  );
}
