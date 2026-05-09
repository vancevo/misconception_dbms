'use client';

import { useState, useEffect, useRef } from 'react';

// === DATA HẰNG SỐ ===
const COLORS = ['#06B6D4','#D946EF','#84CC16','#F59E0B','#EF4444','#8B5CF6'];

const configs = [
  {name:'C1: A+KMeans',nmi:0.32,color:'#475569'},
  {name:'C2: A+HDBSCAN',nmi:0.48,color:'#475569'},
  {name:'C3: A+BERTopic',nmi:0.49,color:'#475569'},
  {name:'C4: B+KMeans',nmi:0.45,color:'#475569'},
  {name:'C5: B+HDBSCAN',nmi:0.62,color:'#06B6D4'},
  {name:'C6: B+BERTopic ★',nmi:0.63,color:'#D946EF'},
  {name:'C7: C+KMeans',nmi:0.41,color:'#475569'},
  {name:'C8: C+HDBSCAN',nmi:0.57,color:'#475569'},
  {name:'C9: C+BERTopic',nmi:0.58,color:'#475569'},
];

// Hàm random tạo điểm
const rng = (() => { let s = 42; return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; }; })();
const centers = [[120,120],[360,90],[220,280],[440,250],[150,380],[390,370]];

export default function Home() {
  const [activeTab, setActiveTab] = useState('viz');
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [tooltip, setTooltip] = useState({ display: false, x: 0, y: 0, content: '' });

  // Dữ liệu Động (Kéo từ Memgraph API)
  const [clusters, setClusters] = useState<any[]>([]);
  const [allPoints, setAllPoints] = useState<any[]>([]);
  
  // API States
  const [syncing, setSyncing] = useState(false);
  const [graphLoading, setGraphLoading] = useState(true);
  const [hfData, setHfData] = useState<any[]>([]);
  const [hfLoading, setHfLoading] = useState(true);

  // Prediction States
  const [studentAnswer, setStudentAnswer] = useState('');
  const [predictResult, setPredictResult] = useState<any>(null);
  const [predicting, setPredicting] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Fetch từ API Graph (Memgraph)
  const fetchGraphData = async () => {
    setGraphLoading(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';
      const response = await fetch(`${apiUrl}/api/graph/misconceptions`);
      const result = await response.json();
      
      if (result.status === 'success' && result.data) {
        // Gán màu sắc cho từng cluster
        const coloredClusters = result.data.map((c: any, i: number) => ({
          ...c,
          color: COLORS[i % COLORS.length]
        }));
        setClusters(coloredClusters);
      } else {
        setClusters([]);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setGraphLoading(false);
    }
  };

  // Tính toán lại Scatter Points mỗi khi Clusters thay đổi
  useEffect(() => {
    if (clusters.length === 0) {
      setAllPoints([]);
      return;
    }
    const points: any[] = [];
    clusters.forEach((c, i) => {
      const [cx, cy] = centers[i % centers.length];
      const n = c.size;
      for(let j=0; j<n; j++) {
        const a = rng() * Math.PI * 2, d = rng() * 55;
        points.push({ x: cx + Math.cos(a)*d, y: cy + Math.sin(a)*d, color: c.color, cid: i });
      }
    });
    setAllPoints(points);
  }, [clusters]);

  // Vẽ Scatter Plot
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#0F172A';
    ctx.beginPath();
    ctx.roundRect(0, 0, canvas.width, canvas.height, 12);
    ctx.fill();

    allPoints.forEach(p => {
      const faded = selectedCluster !== null && p.cid !== selectedCluster;
      ctx.globalAlpha = faded ? 0.1 : 0.85;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.fill();
    });
    ctx.globalAlpha = 1;
  }, [allPoints, selectedCluster, activeTab]);

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    
    let found = null;
    for (const p of allPoints) {
      if (Math.hypot(p.x - mx, p.y - my) < 8) {
        found = p;
        break;
      }
    }

    if (found) {
      const c = clusters[found.cid];
      setTooltip({
        display: true,
        x: e.clientX + 12,
        y: e.clientY - 10,
        content: `<strong style="color:${c.color}">${c.label}</strong><br><span style="color:#94A3B8;font-size:0.75rem">Click để xem chi tiết</span>`
      });
    } else {
      setTooltip(prev => ({ ...prev, display: false }));
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    
    let found = null;
    for (const p of allPoints) {
      if (Math.hypot(p.x - mx, p.y - my) < 8) {
        found = p;
        break;
      }
    }
    if (found) {
      setSelectedCluster(found.cid);
    }
  };

  useEffect(() => {
    fetchGraphData();
    const fetchHuggingFacePreview = async () => {
      setHfLoading(true);
      try {
        const response = await fetch('https://datasets-server.huggingface.co/rows?dataset=vancevo%2Fmisconception_mining&config=default&split=train&offset=0&length=2');
        if (response.ok) {
          const data = await response.json();
          setHfData(data.rows.map((r: any) => r.row));
        }
      } catch (err) { } finally { setHfLoading(false); }
    };
    fetchHuggingFacePreview();
  }, []);

  const handleSyncBackend = async () => {
    setSyncing(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';
      const response = await fetch(`${apiUrl}/api/dataset/sync`, { method: 'POST' });
      const result = await response.json();
      if (!response.ok) {
        alert(`❌ Lỗi đồng bộ:\n${result.message}`);
      } else {
        alert(`✅ Đồng bộ thành công!\n${result.message}`);
        fetchGraphData(); // Reload Graph Data sau khi Sync
      }
    } catch (err) {
      alert('Không thể kết nối đến Local Backend. Hãy chạy python backend/app.py');
    } finally {
      setSyncing(false);
    }
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!studentAnswer.trim()) return;
    setPredicting(true);
    setPredictResult(null);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';
      const response = await fetch(`${apiUrl}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ student_answer: studentAnswer })
      });
      const result = await response.json();
      setPredictResult(result);
    } catch (err) {
      setPredictResult({ status: 'error', message: 'Lỗi kết nối đến Backend' });
    } finally {
      setPredicting(false);
    }
  };

  const activeClusterData = selectedCluster !== null && clusters.length > 0 ? clusters[selectedCluster] : null;

  return (
    <>
      <div className="hero">
        <h1>🔬 Misconception Mining</h1>
        <p>Hệ thống tự động phát hiện <strong style={{color:'#06B6D4'}}>mẫu sai lầm</strong> trong câu trả lời sinh viên bằng AI.</p>
        <div className="stats">
          <div className="stat"><div className="stat-num">{clusters.reduce((acc, c) => acc + c.size, 0) || 0}</div><div className="stat-label">Câu trả lời</div></div>
          <div className="stat"><div className="stat-num" style={{color:'#D946EF'}}>9</div><div className="stat-label">Cấu hình</div></div>
          <div className="stat"><div className="stat-num" style={{color:'#84CC16'}}>{clusters.length}</div><div className="stat-label">Nhóm lỗi sai</div></div>
        </div>
      </div>

      <div className="main-container">
        <div className="tab-bar">
          <button className={`tab ${activeTab === 'viz' ? 'active' : ''}`} onClick={() => setActiveTab('viz')}>🗺️ Bản đồ Memgraph</button>
          <button className={`tab ${activeTab === 'compare' ? 'active' : ''}`} onClick={() => setActiveTab('compare')}>📊 So sánh Cấu hình</button>
          <button className={`tab ${activeTab === 'answers' ? 'active' : ''}`} onClick={() => setActiveTab('answers')}>💬 Câu trả lời mẫu</button>
          <button className={`tab ${activeTab === 'predict' ? 'active' : ''}`} onClick={() => setActiveTab('predict')}>🤖 Thử nghiệm AI</button>
          <button className={`tab ${activeTab === 'api' ? 'active' : ''}`} onClick={() => setActiveTab('api')}>⚡ Local Backend & Sync</button>
        </div>

        {activeTab === 'viz' && (
          <div>
            {clusters.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', background: '#1E293B', borderRadius: '12px' }}>
                <h3 style={{ color: '#F59E0B' }}>Database Memgraph Đang Trống!</h3>
                <p style={{ marginTop: '10px' }}>Vui lòng qua tab "Local Backend & Sync" để khởi tạo Graph Database.</p>
              </div>
            ) : (
              <>
                <div className="grid2">
                  <div className="card">
                    <div className="section-title">UMAP — Bản đồ lỗi sai 2D</div>
                    <canvas 
                      ref={canvasRef} 
                      width="500" height="380" 
                      style={{ width: '100%', height: 'auto', maxWidth: '500px' }}
                      onMouseMove={handleCanvasMouseMove}
                      onMouseLeave={() => setTooltip(p => ({...p, display: false}))}
                      onClick={handleCanvasClick}
                    />
                    <div className="legend">
                      {clusters.map((c, idx) => (
                        <div key={c.id} className="legend-item" onClick={() => setSelectedCluster(prev => prev === idx ? null : idx)}>
                          <div className="dot" style={{background: c.color}}></div>
                          <span style={{fontSize: '0.8rem', color: '#94A3B8'}}>{c.label}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="card">
                    {!activeClusterData ? (
                      <div style={{color:'#475569', textAlign:'center', padding:'60px 0'}}>👆 Click vào một điểm hoặc chọn nhóm bên trái</div>
                    ) : (
                      <>
                        <div className="section-title">Chi tiết nhóm</div>
                        <div style={{fontSize:'1.2rem', fontWeight:700, color:activeClusterData.color}}>{activeClusterData.label}</div>
                        <div style={{fontSize:'0.8rem', color:'#64748B', marginBottom:'16px'}}>
                          {activeClusterData.size} câu trả lời · Độ kết dính: {activeClusterData.cohesion}
                        </div>
                        <div className="keywords">
                          {activeClusterData.keywords.map((kw: string) => <span key={kw} className="kw">{kw}</span>)}
                        </div>
                        <div className="answers" style={{marginTop: '16px'}}>
                          {activeClusterData.answers.slice(0,3).map((a: string, i: number) => (
                            <div key={i} className="answer-card">"{a}"</div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* TAB 2 & 3: Dữ liệu tĩnh như cũ */}
        {activeTab === 'compare' && (
          <div className="card">
             <div className="section-title">So sánh 9 cấu hình (NMI)</div>
             <div className="bar-chart">
               {configs.map((c, idx) => {
                 const pct = Math.round((c.nmi / 0.65) * 100);
                 return (
                   <div key={idx} className="bar-row">
                     <div className="bar-label" style={{fontSize: '0.75rem'}}>{c.name}</div>
                     <div className="bar-track"><div className="bar-fill" style={{width: `${pct}%`, background: c.color}}>{c.nmi}</div></div>
                   </div>
                 );
               })}
             </div>
          </div>
        )}

        {activeTab === 'answers' && (
          <div className="grid2">
            <div className="card">
              <div className="section-title">Chọn nhóm lỗi sai</div>
              <div className="cluster-list">
                {clusters.map((c, idx) => (
                  <div key={c.id} className={`cluster-row ${selectedCluster === idx ? 'active' : ''}`} onClick={() => setSelectedCluster(idx)}>
                    <div className="cluster-name"><span className="dot" style={{background: c.color}}></span>{c.label}</div>
                    <div className="cluster-meta">{c.size} câu trả lời · từ khóa: {c.keywords.slice(0,3).join(', ')}</div>
                  </div>
                ))}
              </div>
            </div>
            <div className="card">
              <div className="section-title">Câu trả lời trong nhóm</div>
              <div className="answers">
                {!activeClusterData ? (
                  <div style={{color: '#475569', textAlign: 'center', padding: '40px 0'}}>Chọn một nhóm bên trái</div>
                ) : (
                  activeClusterData.answers.map((a: string, i: number) => (
                    <div key={i} className="answer-card">
                      "{a}"
                      <div className="label">Nhóm: <strong style={{color: activeClusterData.color}}>{activeClusterData.label}</strong></div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB PREDICT: THỬ NGHIỆM AI */}
        {activeTab === 'predict' && (
          <div className="card" style={{ maxWidth: '600px', margin: '0 auto' }}>
             <div className="section-title">Thử nghiệm Mô hình AI</div>
             <p style={{ color: '#94A3B8', marginBottom: '20px', fontSize: '0.9rem' }}>
               Nhập câu trả lời của sinh viên vào bên dưới để hệ thống SBERT phân tích lỗi sai (Misconception).
             </p>
             <form onSubmit={handlePredict} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
               <textarea 
                 value={studentAnswer}
                 onChange={(e) => setStudentAnswer(e.target.value)}
                 placeholder="Ví dụ: Lực và năng lượng là một, chỉ khác đơn vị đo..."
                 style={{ 
                   width: '100%', minHeight: '100px', padding: '12px', 
                   borderRadius: '8px', background: '#0F172A', 
                   border: '1px solid #334155', color: '#fff', fontSize: '1rem',
                   fontFamily: 'inherit', resize: 'vertical'
                 }}
               />
               <button 
                 type="submit" 
                 disabled={predicting} 
                 className="button"
                 style={{ alignSelf: 'flex-start', background: '#D946EF', border: 'none', padding: '10px 20px', color: '#fff', borderRadius: '8px', cursor: predicting ? 'not-allowed' : 'pointer' }}
               >
                 {predicting ? 'Đang phân tích...' : '✨ Phân tích lỗi sai'}
               </button>
             </form>

             {predictResult && (
               <div style={{ marginTop: '24px', padding: '16px', borderRadius: '8px', background: predictResult.status === 'success' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)', border: `1px solid ${predictResult.status === 'success' ? '#10B981' : '#EF4444'}` }}>
                 {predictResult.status === 'success' ? (
                   <>
                     <div style={{ fontSize: '0.85rem', color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '8px' }}>Kết quả dự đoán</div>
                     <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: '#10B981', marginBottom: '8px' }}>
                       {predictResult.prediction.cluster_label}
                     </div>
                     <div style={{ fontSize: '0.9rem', color: '#CBD5E1' }}>
                       Nhóm ID: <strong style={{color: '#fff'}}>{predictResult.prediction.cluster_id}</strong>
                       <span style={{margin: '0 10px'}}>|</span>
                       Độ tự tin: <strong style={{color: '#fff'}}>{(predictResult.prediction.confidence_score * 100).toFixed(1)}%</strong>
                     </div>
                     {predictResult.message && (
                       <div style={{ marginTop: '12px', fontSize: '0.85rem', color: '#F59E0B' }}>
                         ⚠️ {predictResult.message}
                       </div>
                     )}
                   </>
                 ) : (
                   <div style={{ color: '#EF4444' }}>❌ {predictResult.message}</div>
                 )}
               </div>
             )}
          </div>
        )}

        {/* TAB 4: API TÍCH HỢP HỆ THỐNG */}
        {activeTab === 'api' && (
          <div className="grid2">
            {/* API Status */}
            <section className="card">
              <h2 className="card-title" style={{fontSize: '1.1rem', marginBottom: '1rem'}}>Local Backend Control</h2>
              <ul className="data-list">
                <li className="data-item">
                  <span>Memgraph Database</span>
                  <span className="badge" style={{background: '#10b981'}}>Running on Port 7687</span>
                </li>
                <li className="data-item">
                  <span>Python Flask Backend</span>
                  <span className="badge" style={{background: '#3b82f6'}}>http://localhost:5001</span>
                </li>
              </ul>
              <button className="button" onClick={handleSyncBackend} disabled={syncing}>
                {syncing ? 'Đang tạo Knowledge Graph...' : 'Sync Data to Memgraph'}
              </button>
            </section>
          </div>
        )}
      </div>

      {tooltip.display && (
        <div className="tooltip-custom" style={{ display: 'block', left: tooltip.x, top: tooltip.y }} dangerouslySetInnerHTML={{ __html: tooltip.content }} />
      )}
    </>
  );
}
