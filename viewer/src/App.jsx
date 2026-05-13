import React, { useState, useEffect, useRef } from 'react';
import { 
  Search, FileText, CheckCircle2, AlertCircle, ExternalLink, 
  ChevronRight, Hash, ChevronLeft, List 
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [data, setData] = useState({ documents: [], results: {} });
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [docContent, setDocContent] = useState('');
  const [activeCitations, setActiveCitations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
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
        console.error("Fetch error:", err);
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

  // Logic to map chunk index to page range (based on 5-page chunks with 1-page overlap)
  const getPagesForChunk = (chunkIdx) => {
    const CHUNK_SIZE = 5;
    const OVERLAP = 1;
    const startPage = chunkIdx * (CHUNK_SIZE - OVERLAP) + 1;
    const endPage = startPage + CHUNK_SIZE - 1;
    return { start: startPage, end: endPage };
  };

  const handleJump = (chunkIdx) => {
    const range = getPagesForChunk(chunkIdx);
    setActiveCitations([chunkIdx]);
    
    // Scroll to the first page of the chunk
    setTimeout(() => {
      const el = document.getElementById(`page-${range.start}`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 50);
  };

  const renderContent = () => {
    if (!docContent) return <div className="p-12 text-slate-400">Select a document to begin...</div>;

    // Split by page markers: === Page X ===
    const pageParts = docContent.split(/=== Page (\d+) ===/);
    if (pageParts.length <= 1) {
       return <pre className="whitespace-pre-wrap p-12">{docContent}</pre>;
    }

    const elements = [];
    // pageParts[0] is header info
    if (pageParts[0].trim()) {
        elements.push(<div key="header" className="p-12 border-b border-slate-100 text-slate-400 italic">{pageParts[0]}</div>);
    }

    for (let i = 1; i < pageParts.length; i += 2) {
      const pageNum = parseInt(pageParts[i]);
      const pageText = pageParts[i + 1];
      
      // Determine if this page is part of the currently active chunk(s)
      const isHighlighted = activeCitations.some(idx => {
        const range = getPagesForChunk(idx);
        return pageNum >= range.start && pageNum <= range.end;
      });

      elements.push(
        <div 
          key={pageNum} 
          id={`page-${pageNum}`}
          className={`relative mb-12 p-12 transition-all rounded-3xl border ${
            isHighlighted 
              ? 'bg-blue-50/50 border-blue-200 ring-4 ring-blue-500/5 shadow-2xl' 
              : 'bg-white border-slate-100 shadow-sm'
          }`}
        >
          <div className="absolute -top-4 left-8 px-4 py-1 bg-slate-900 text-white text-[10px] font-black rounded-full uppercase tracking-widest shadow-lg">
            Page {pageNum}
          </div>
          <pre className="whitespace-pre-wrap font-mono text-[13px] leading-relaxed text-slate-700">
            {pageText.trim()}
          </pre>
        </div>
      );
    }

    return <div className="p-12 space-y-8">{elements}</div>;
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-white text-slate-800">
        <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-6"></div>
        <p className="text-lg font-semibold animate-pulse tracking-wide">Syncing Chunks...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-white p-12 text-center">
        <AlertCircle size={48} className="text-red-500 mb-6" />
        <h2 className="text-2xl font-bold text-slate-900 mb-3">Connection Error</h2>
        <p className="text-slate-500 max-w-md text-lg">{error}</p>
        <button onClick={() => window.location.reload()} className="mt-8 px-8 py-3 bg-blue-600 text-white rounded-xl font-semibold">Retry</button>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-white overflow-hidden text-slate-800">
      {/* Sidebar */}
      <div className="w-80 bg-slate-50 border-r border-slate-200 flex flex-col">
        <div className="p-8 border-b border-slate-200 bg-white shadow-sm">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2.5 rounded-xl shadow-lg shadow-blue-600/30">
              <FileText size={20} className="text-white" />
            </div>
            <div>
              <h1 className="text-lg font-black text-slate-900 tracking-tight leading-none">BIDPANION</h1>
              <p className="text-[10px] text-slate-400 mt-1 uppercase font-bold tracking-[0.2em]">Chunk Visualizer</p>
            </div>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-6 space-y-3">
          {data.documents.map(doc => (
            <button
              key={doc}
              onClick={() => setSelectedDoc(doc)}
              className={`w-full flex items-center gap-3 px-5 py-4 rounded-2xl transition-all text-left ${
                selectedDoc === doc 
                  ? 'bg-blue-600 text-white shadow-xl shadow-blue-600/20' 
                  : 'hover:bg-slate-200/50 text-slate-600'
              }`}
            >
              <FileText size={18} className={selectedDoc === doc ? 'text-white' : 'text-slate-400'} />
              <span className="text-sm font-bold truncate">{doc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Document View */}
      <div className="flex-1 overflow-y-auto bg-slate-50/30 scroll-smooth shadow-inner" ref={docRef}>
        <div className="sticky top-0 z-10 bg-white/80 backdrop-blur-xl px-10 py-6 border-b border-slate-200 flex justify-between items-center shadow-sm">
          <div className="flex items-center gap-3">
             <div className="flex items-center gap-2 text-[11px] font-black text-slate-400 uppercase tracking-widest">
               <FileText size={14} /> Document
             </div>
             <ChevronRight size={14} className="text-slate-300" />
             <span className="text-sm font-bold text-slate-900">{selectedDoc}</span>
          </div>
        </div>
        <div className="max-w-4xl mx-auto">
          {renderContent()}
        </div>
      </div>

      {/* Analysis Results Panel */}
      <div className="w-[500px] bg-white border-l border-slate-200 overflow-y-auto shadow-2xl z-20">
        <div className="sticky top-0 z-10 bg-white/90 backdrop-blur-md px-8 py-7 border-b border-slate-200">
          <h2 className="text-sm font-black uppercase tracking-widest text-slate-900 flex items-center gap-2">
            <List size={16} className="text-blue-600" /> Analysis Results
          </h2>
        </div>
        <div className="p-8 space-y-10">
          {Object.entries(data.results).map(([section, value]) => {
            if (section === 'citations') return null;
            const isObject = typeof value === 'object' && value !== null && !Array.isArray(value);
            return (
              <div key={section} className="space-y-6">
                <div className="flex items-center gap-4">
                  <h3 className="text-[11px] font-black text-slate-400 uppercase tracking-[0.2em]">{section}</h3>
                  <div className="h-[1px] flex-1 bg-slate-100"></div>
                </div>
                <div className="space-y-4">
                {isObject ? (
                   Object.entries(value).map(([field, val]) => (
                     <FieldCard key={field} title={field} value={val} citationStr={data.results.citations?.sources[`${field}__quelle`]} onJump={handleJump} getPages={getPagesForChunk} />
                   ))
                ) : (
                  <FieldCard title={section} value={value} citationStr={data.results.citations?.sources[`${section}__quelle`]} onJump={handleJump} getPages={getPagesForChunk} />
                )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function FieldCard({ title, value, citationStr, onJump, getPages }) {
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
  const currentRange = citations.length > 0 ? getPages(citations[activeIndex]) : null;
  
  return (
    <div 
      className={`group p-6 rounded-3xl transition-all border shadow-sm ${
        citations.length > 0 ? 'hover:border-blue-500/50 cursor-pointer bg-white' : 'bg-slate-50 border-slate-100 opacity-80'
      }`}
      onClick={() => citations.length > 0 && onJump(citations[activeIndex])}
    >
      <div className="flex justify-between items-start mb-6">
        <h4 className="text-[11px] font-black text-slate-400 uppercase tracking-wider group-hover:text-blue-600 transition-colors">{title}</h4>
        
        {citations.length > 0 && (
          <div className="flex items-center gap-1 bg-slate-100 p-1.5 rounded-xl">
             <button onClick={handlePrev} className="p-1.5 hover:bg-white rounded-lg transition-colors text-slate-400 hover:text-blue-600"><ChevronLeft size={14} /></button>
             <div className="px-3 py-0.5 text-[10px] font-black text-slate-600 text-center min-w-[50px]">
               {activeIndex + 1} / {citations.length}
             </div>
             <button onClick={handleNext} className="p-1.5 hover:bg-white rounded-lg transition-colors text-slate-400 hover:text-blue-600"><ChevronRight size={14} /></button>
          </div>
        )}
      </div>

      <div className="text-[14px] text-slate-700 leading-relaxed font-medium">
        {Array.isArray(value) ? (
          <ul className="space-y-3">
            {value.map((item, i) => (
              <li key={i} className="flex gap-4 items-start">
                <div className="w-2 h-2 rounded-full bg-blue-500 mt-2 shrink-0"></div>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        ) : typeof value === 'object' && value !== null ? (
          <div className="space-y-4">
            {Object.entries(value).map(([k, v]) => (
              <div key={k} className="bg-slate-50/50 p-4 rounded-2xl border border-slate-100">
                <span className="text-[10px] text-slate-400 font-black uppercase block mb-1 tracking-widest">{k}</span>
                <span className="text-slate-800">{typeof v === 'object' ? JSON.stringify(v) : (v || 'Not found')}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="whitespace-pre-wrap">{value || 'No information found'}</p>
        )}
      </div>

      <div className="mt-8 pt-6 border-t border-slate-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isFound ? (
            <div className="flex items-center gap-2 px-4 py-1.5 bg-emerald-50 rounded-full border border-emerald-100">
              <CheckCircle2 size={14} className="text-emerald-500" />
              <span className="text-[10px] text-emerald-600 font-black uppercase tracking-widest">Verified</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-4 py-1.5 bg-amber-50 rounded-full border border-amber-100">
              <AlertCircle size={14} className="text-amber-500" />
              <span className="text-[10px] text-amber-600 font-black uppercase tracking-widest">Absent</span>
            </div>
          )}
        </div>
        
        {currentRange && (
          <div className="flex items-center gap-2 px-4 py-1.5 bg-blue-50 rounded-full border border-blue-100 text-blue-600 text-[10px] font-black uppercase tracking-widest">
            Pages {currentRange.start}–{currentRange.end} <Hash size={12} />
          </div>
        )}
      </div>
    </div>
  );
}
