import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../components/ui';
import { Brain, ArrowRight, Check } from 'lucide-react';

const Landing = () => {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen relative overflow-hidden flex flex-col">
            {/* Background */}
            <div className="absolute inset-0 bg-slate-950">
                <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-cyan-600/10 rounded-full blur-[100px] -translate-y-1/2 translate-x-1/2"></div>
                <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-purple-600/10 rounded-full blur-[100px] translate-y-1/3 -translate-x-1/3"></div>
            </div>

            {/* Navbar */}
            <nav className="container mx-auto px-6 py-8 relative z-10 flex justify-between items-center">
                <div className="flex items-center gap-3">
                    <div className="bg-gradient-to-r from-cyan-500 to-blue-600 p-2 rounded-lg shadow-lg shadow-cyan-500/20">
                        <Brain className="w-6 h-6 text-white" />
                    </div>
                    <span className="text-xl font-bold text-white tracking-tight">NeuroDetect</span>
                </div>
                <div className="flex items-center gap-4">
                    <Button variant="ghost" onClick={() => navigate('/login')} className="text-slate-300 hover:text-white">
                        Sign In
                    </Button>
                    <Button onClick={() => navigate('/login')} variant="primary" className="shadow-cyan-500/20">
                        Get Started
                    </Button>
                </div>
            </nav>

            {/* Hero */}
            <main className="flex-1 container mx-auto px-6 flex flex-col items-center justify-center text-center relative z-10 -mt-20">
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-900/50 border border-slate-700/50 mb-8 animate-fade-in backdrop-blur-sm">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                    <span className="text-xs font-medium text-cyan-400 uppercase tracking-widest">AI-Powered Diagnosis v2.0</span>
                </div>

                <h1 className="text-5xl md:text-7xl font-bold text-white mb-8 leading-tight tracking-tight animate-scale-in">
                    Early Detection <br />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
                        Changes Everything
                    </span>
                </h1>

                <p className="text-xl text-slate-400 max-w-2xl mb-12 leading-relaxed animate-fade-in delay-100">
                    Leveraging advanced computer vision and voice biomarker analysis to identify
                    early signs of Parkinson's disease with clinical-grade accuracy.
                </p>

                <div className="flex flex-col sm:flex-row gap-4 animate-fade-in delay-200">
                    <Button
                        onClick={() => navigate('/login')}
                        className="py-4 px-8 text-lg bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 shadow-xl shadow-cyan-900/20"
                    >
                        Start Assessment <ArrowRight className="ml-2" />
                    </Button>
                    <Button
                        onClick={() => navigate('/login')}
                        variant="secondary"
                        className="py-4 px-8 text-lg bg-slate-900/50 hover:bg-slate-800 border-slate-700 backdrop-blur-md"
                    >
                        Login to Dashboard
                    </Button>
                </div>

                {/* Features */}
                <div className="mt-24 grid md:grid-cols-3 gap-8 text-left max-w-5xl animate-fade-in delay-300">
                    {[
                        { title: "Spiral Analysis", desc: "Detects micro-tremors in motor control patterns." },
                        { title: "Vocal Biomarkers", desc: "Analyzes frequency and jitter in voice recordings." },
                        { title: "Instant Results", desc: "Get comprehensive diagnostic reports in seconds." }
                    ].map((feature, i) => (
                        <div key={i} className="flex gap-4 p-6 rounded-2xl bg-slate-900/30 border border-slate-800 hover:bg-slate-800/50 transition-colors">
                            <div className="mt-1 bg-cyan-500/20 w-6 h-6 rounded-full flex items-center justify-center shrink-0">
                                <Check className="w-3 h-3 text-cyan-400" />
                            </div>
                            <div>
                                <h3 className="text-white font-bold mb-1">{feature.title}</h3>
                                <p className="text-slate-500 text-sm">{feature.desc}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </main>
        </div>
    );
};

export default Landing;
