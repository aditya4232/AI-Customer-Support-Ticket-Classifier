import React, { useState } from 'react';
import axios from 'axios';
import { Sparkles, Send, AlertTriangle, CheckCircle2, Server, Clock, AlertCircle } from 'lucide-react';

export default function App() {
  const [ticketText, setTicketText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!ticketText.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post('http://localhost:8000/api/predict', {
        ticket_text: ticketText
      });
      setResult(response.data.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while connecting to the server.');
    } finally {
      setLoading(false);
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'high': return 'bg-red-500/10 text-red-600 border-red-200';
      case 'medium': return 'bg-amber-500/10 text-amber-600 border-amber-200';
      case 'low': return 'bg-emerald-500/10 text-emerald-600 border-emerald-200';
      default: return 'bg-slate-100 text-slate-600 border-slate-200';
    }
  };

  const getPriorityIcon = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'high': return <AlertTriangle className="w-5 h-5 text-red-500 mr-2" />;
      case 'medium': return <AlertCircle className="w-5 h-5 text-amber-500 mr-2" />;
      case 'low': return <CheckCircle2 className="w-5 h-5 text-emerald-500 mr-2" />;
      default: return <Clock className="w-5 h-5 text-slate-500 mr-2" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 text-slate-900 font-sans selection:bg-blue-200">
      {/* Navbar */}
      <nav className="fixed w-full top-0 backdrop-blur-xl bg-white/70 border-b border-white/20 z-50 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-2">
              <div className="bg-blue-600 text-white p-2 rounded-xl shadow-lg shadow-blue-600/20">
                <Server className="w-5 h-5" />
              </div>
              <span className="font-bold text-xl tracking-tight text-slate-800">SupportSync AI</span>
            </div>
            <div className="text-sm font-medium text-slate-500">
              Telecom Domain Classifier
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="pt-32 pb-16 px-4 sm:px-6 lg:px-8 max-w-6xl mx-auto flex flex-col lg:flex-row gap-8 items-start">
        
        {/* Left Column - Input area */}
        <div className="w-full lg:w-1/2 flex flex-col space-y-8">
          <div>
            <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-slate-900 mb-4 inline-flex items-center gap-3">
              Intelligent Ticket <br/> Classification
              <Sparkles className="w-8 h-8 text-blue-500 animate-pulse mt-2" />
            </h1>
            <p className="text-lg text-slate-600 leading-relaxed max-w-lg">
              Paste your customer support query below. Our AI model will automatically analyze and classify it into the correct 
              department while assigning an exact priority level.
            </p>
          </div>

          <div className="bg-white rounded-3xl p-2 shadow-xl shadow-slate-200/50 border border-slate-100/50 relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-transparent opacity-0 group-focus-within:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
            <form onSubmit={handlePredict} className="relative bg-white rounded-2xl p-4 sm:p-6 shadow-sm border border-slate-100">
              <label htmlFor="ticket" className="block text-sm font-semibold text-slate-700 mb-2">
                Customer Message
              </label>
              <textarea
                id="ticket"
                value={ticketText}
                onChange={(e) => setTicketText(e.target.value)}
                placeholder="E.g., I was charged twice for my subscription this month. Please refund..."
                className="w-full h-48 sm:h-64 p-4 text-slate-700 bg-slate-50 border border-slate-200 rounded-xl focus:ring-4 focus:ring-blue-500/10 focus:border-blue-500 transition-all resize-none outline-none mb-4"
              />
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400 font-medium">Powered by Scikit-Learn (MultinomialNB)</span>
                <button
                  type="submit"
                  disabled={loading || !ticketText.trim()}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-xl font-semibold tracking-wide shadow-lg shadow-blue-600/20 transition-all hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:hover:translate-y-0 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {loading ? (
                    <span className="flex items-center gap-2">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Analyzing...
                    </span>
                  ) : (
                    <>
                      <span>Classify Ticket</span>
                      <Send className="w-4 h-4 ml-1" />
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
          
          {error && (
            <div className="bg-red-50 text-red-600 p-4 rounded-xl text-sm border border-red-100 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right Column - Results area */}
        <div className="w-full lg:w-1/2 flex flex-col relative">
          {!result ? (
            <div className="h-full min-h-[400px] rounded-3xl border-2 border-dashed border-slate-200 flex flex-col items-center justify-center text-center p-8 bg-slate-50/50">
              <div className="w-20 h-20 bg-blue-50 text-blue-500 rounded-full flex items-center justify-center mb-6 shadow-inner">
                <Server className="w-10 h-10 opacity-50" />
              </div>
              <h3 className="text-xl font-bold text-slate-700 mb-2">Awaiting Telemetry</h3>
              <p className="text-slate-500 max-w-sm">
                Submit a customer query to see real-time classification, routing category, and predicted priority score.
              </p>
            </div>
          ) : (
            <div className="space-y-6 transition-all duration-700 ease-out">
              <div className="bg-white p-8 rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-indigo-500"></div>
                
                <h2 className="text-xs font-bold tracking-widest text-slate-400 uppercase mb-8">Analysis Results</h2>
                
                <div className="grid grid-cols-2 gap-6 mb-8">
                  {/* Category Card */}
                  <div className="bg-slate-50 rounded-2xl p-5 border border-slate-100">
                    <span className="text-sm font-semibold text-slate-500 mb-1 block">Assigned Category</span>
                    <div className="text-2xl font-bold text-slate-900 flex items-center gap-2">
                       {result.category}
                    </div>
                    <div className="mt-2 text-xs font-medium text-slate-400">
                      Confidence: {(result.category_confidence * 100).toFixed(1)}%
                    </div>
                  </div>

                  {/* Priority Card */}
                  <div className={`rounded-2xl p-5 border ${getPriorityColor(result.priority)}`}>
                    <span className="text-sm font-semibold opacity-80 mb-1 block">Priority Level</span>
                    <div className="text-2xl font-bold flex items-center">
                       {getPriorityIcon(result.priority)}
                       {result.priority}
                    </div>
                    <div className="mt-2 text-xs font-medium opacity-70">
                      Confidence: {(result.priority_confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Reasoning Section */}
                <div className="mt-6 border-t border-slate-100 pt-6">
                  <h3 className="text-sm font-bold text-slate-800 mb-3 flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-blue-500" />
                    AI Reasoning
                  </h3>
                  <div className="bg-blue-50/50 text-slate-700 p-4 rounded-xl text-sm leading-relaxed border border-blue-100/50">
                    {result.reasoning?.category_reasoning || "Reasoning unavailable."}
                  </div>
                </div>
                
              </div>
            </div>
          )}
        </div>
        
      </main>
    </div>
  );
}
