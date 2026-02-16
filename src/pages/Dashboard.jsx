import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, Card } from '../components/ui';
import { Activity, Mic, LogOut, ArrowRight, Brain, Zap, Shield } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Dashboard = () => {
    const navigate = useNavigate();
    const { logout, user } = useAuth();

    const handleLogout = () => {
        logout();
        navigate('/');
    };

    return (
        <div className="min-h-screen p-6 animate-fade-in relative overflow-hidden">
            {/* Background Accents */}
            <div className="absolute top-0 left-0 w-[500px] h-[500px] bg-cyan-500/10 rounded-full blur-[128px] pointer-events-none -translate-x-1/2 -translate-y-1/2"></div>
            <div className="absolute bottom-0 right-0 w-[500px] h-[500px] bg-blue-600/10 rounded-full blur-[128px] pointer-events-none translate-x-1/2 translate-y-1/2"></div>

            <nav className="flex justify-between items-center mb-16 container mx-auto relative z-10">
                <div className="flex items-center gap-3">
                    <div className="bg-gradient-to-tr from-cyan-500 to-blue-600 p-2 rounded-lg">
                        <Brain className="w-6 h-6 text-white" />
                    </div>
                    <h1 className="text-2xl font-bold text-white tracking-tight">NeuroDetect</h1>
                </div>
                <div className="flex items-center gap-6">
                    <span className="text-slate-400 text-sm hidden sm:block">Welcome, <span className="text-white font-medium">{user?.username}</span></span>
                    <Button variant="ghost" onClick={handleLogout} className="flex items-center gap-2 text-sm">
                        <LogOut size={16} /> <span className="hidden sm:inline">Logout</span>
                    </Button>
                </div>
            </nav>

            <div className="container mx-auto px-4 flex flex-col items-center justify-center min-h-[60vh] relative z-10">
                <div className="text-center max-w-3xl mb-16 animate-scale-in">
                    <h2 className="text-5xl font-bold mb-6 text-white tracking-tight">Clinical-Grade Assessment</h2>
                    <p className="text-slate-400 text-xl leading-relaxed">
                        Our advanced multi-modal system combines
                        <span className="text-cyan-400"> motor analysis</span> and
                        <span className="text-purple-400"> vocal biomarkers</span>
                        to detect early signs of Parkinson's disease with high precision.
                    </p>
                </div>

                <div className="grid md:grid-cols-3 gap-8 w-full max-w-5xl mb-16">
                    <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl flex flex-col items-center text-center">
                        <Activity className="w-10 h-10 text-cyan-400 mb-4" />
                        <h3 className="text-white font-bold mb-2">Motor Control</h3>
                        <p className="text-slate-500 text-sm">Spiral graphometry analysis detects micro-tremors.</p>
                    </div>
                    <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl flex flex-col items-center text-center">
                        <Mic className="w-10 h-10 text-purple-400 mb-4" />
                        <h3 className="text-white font-bold mb-2">Vocal Branding</h3>
                        <p className="text-slate-500 text-sm">Analyzes frequency jitter, shimmer, and entropy.</p>
                    </div>
                    <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl flex flex-col items-center text-center">
                        <Shield className="w-10 h-10 text-green-400 mb-4" />
                        <h3 className="text-white font-bold mb-2">Private & Secure</h3>
                        <p className="text-slate-500 text-sm">Your diagnostic data is encrypted and private.</p>
                    </div>
                </div>

                <Card
                    className="w-full max-w-2xl hover:scale-[1.02] transition-all duration-300 cursor-pointer border-cyan-500/30 hover:border-cyan-500 shadow-2xl shadow-cyan-900/20 group backdrop-blur-xl bg-slate-900/60"
                    onClick={() => navigate('/procedure')}
                >
                    <div className="flex flex-col items-center text-center p-10">
                        <div className="bg-gradient-to-br from-cyan-500/20 to-blue-500/20 p-6 rounded-full mb-6 relative group-hover:bg-cyan-500/30 transition-colors">
                            <Zap className="w-12 h-12 text-cyan-400" />
                        </div>

                        <h3 className="text-2xl font-bold mb-2 text-white group-hover:text-cyan-400 transition-colors">Start New Procedure</h3>
                        <p className="text-slate-400 mb-8">
                            Initialize the guided 2-step diagnostic workflow.
                        </p>

                        <Button className="w-full py-5 text-lg bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 shadow-lg shadow-cyan-900/20">
                            Initialize System <ArrowRight className="ml-2" />
                        </Button>
                    </div>
                </Card>
            </div>
        </div>
    );
};

export default Dashboard;
