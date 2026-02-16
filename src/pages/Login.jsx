import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Button, Card } from '../components/ui';
import { useNavigate } from 'react-router-dom';
import { GoogleLogin } from '@react-oauth/google';
import { Shield, Lock, Brain } from 'lucide-react';

const Login = () => {
    const { googleLogin } = useAuth();
    const navigate = useNavigate();
    const [error, setError] = useState('');

    const handleSuccess = async (credentialResponse) => {
        try {
            if (credentialResponse.credential) {
                const success = await googleLogin(credentialResponse.credential);
                if (success) {
                    navigate('/dashboard');
                } else {
                    setError('Authentication credentials rejected');
                }
            }
        } catch (err) {
            console.error(err);
            setError('System login failed');
        }
    };

    const handleError = () => {
        setError('Secure handshake failed');
    };

    return (
        <div className="min-h-screen flex flex-col items-center justify-center p-4 bg-slate-950 text-slate-50 relative overflow-hidden">
            {/* Background */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-cyan-950/20 to-slate-950 -z-10"></div>

            <div className="mb-8 text-center animate-fade-in">
                <div className="inline-flex items-center gap-3 mb-4">
                    <div className="bg-cyan-500/10 p-3 rounded-xl border border-cyan-500/20">
                        <Brain className="w-8 h-8 text-cyan-400" />
                    </div>
                </div>
                <h1 className="text-2xl font-bold tracking-tight">NeuroDetect <span className="text-cyan-500">AI</span></h1>
                <p className="text-slate-500 text-sm tracking-wide uppercase mt-2">Authorized Access Only</p>
            </div>

            <Card className="w-full max-w-md animate-fade-in shadow-2xl border-slate-800">
                <div className="flex items-center justify-center mb-8">
                    <div className="w-16 h-16 bg-slate-900 rounded-full flex items-center justify-center text-cyan-400 border border-slate-800">
                        <Lock size={24} />
                    </div>
                </div>

                <h2 className="text-xl font-bold text-center mb-2 text-white">Clinician Login</h2>
                <p className="text-slate-400 text-sm text-center mb-8">
                    Securely access patient records and diagnostic tools.
                </p>

                {error && (
                    <div className="mb-6 bg-rose-950/20 border border-rose-900/50 text-rose-400 p-3 rounded-lg text-sm text-center">
                        {error}
                    </div>
                )}

                <div className="flex flex-col gap-4">
                    <div className="flex justify-center">
                        <GoogleLogin
                            onSuccess={handleSuccess}
                            onError={handleError}
                            theme="filled_black"
                            shape="pill"
                            size="large"
                            width="300"
                            text="continue_with"
                        />
                    </div>


                </div>

                <div className="mt-8 pt-6 border-t border-slate-800 text-center">
                    <div className="flex items-center justify-center gap-2 text-xs text-slate-600">
                        <Shield size={12} />
                        <span>256-bit End-to-End Encryption</span>
                    </div>
                </div>
            </Card>
        </div>
    );
};

export default Login;
