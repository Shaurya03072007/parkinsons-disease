import React, { useState, useEffect } from 'react';
import { Button, Card } from '../components/ui';
import { Upload, Activity, Mic, ArrowRight, Brain, CheckCircle, AlertCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Procedure = () => {
    const navigate = useNavigate();

    // State: 'start', 'image', 'voice', 'analyzing_image', 'analyzing_voice', 'result'
    const [step, setStep] = useState('start');

    const [imageFile, setImageFile] = useState(null);
    const [imagePreview, setImagePreview] = useState(null);
    const [imageResult, setImageResult] = useState(null);

    const [voiceFile, setVoiceFile] = useState(null);
    const [voiceResult, setVoiceResult] = useState(null);

    const [error, setError] = useState('');

    // --- Handlers ---

    const handleStart = () => {
        setStep('image');
    };

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImageFile(file);
            setImagePreview(URL.createObjectURL(file));
            setError('');
        }
    };

    const handleVoiceChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setVoiceFile(file);
            setError('');
        }
    };

    const analyzeImage = async () => {
        if (!imageFile) return;

        setStep('analyzing_image');

        // Simulate "video buffer" effect for 3 seconds
        const bufferTime = new Promise(resolve => setTimeout(resolve, 3000));

        const formData = new FormData();
        formData.append('file', imageFile);

        try {
            const apiCall = fetch('http://localhost:8000/predict/image', {
                method: 'POST',
                body: formData,
            });

            const [response] = await Promise.all([apiCall, bufferTime]);

            if (!response.ok) throw new Error('Image analysis failed');

            const data = await response.json();
            setImageResult(data);
            setStep('voice'); // Move to next step
        } catch (err) {
            setError(err.message || 'Analysis failed');
            setStep('image'); // Go back
        }
    };

    const analyzeVoice = async () => {
        if (!voiceFile) return;

        setStep('analyzing_voice');

        // Simulate "video buffer" effect
        const bufferTime = new Promise(resolve => setTimeout(resolve, 3000));

        const formData = new FormData();
        formData.append('file', voiceFile);

        try {
            const apiCall = fetch('http://localhost:8000/predict/voice', {
                method: 'POST',
                body: formData,
            });

            const [response] = await Promise.all([apiCall, bufferTime]);

            if (!response.ok) throw new Error('Voice analysis failed');

            const data = await response.json();
            setVoiceResult(data);
            setStep('result'); // Finished!
        } catch (err) {
            setError(err.message || 'Analysis failed');
            setStep('voice'); // Go back
        }
    };

    const restart = () => {
        setStep('start');
        setImageFile(null);
        setImagePreview(null);
        setImageResult(null);
        setVoiceFile(null);
        setVoiceResult(null);
        setError('');
    };

    // --- Renders ---

    return (
        <div className="min-h-screen p-6 animate-fade-in container mx-auto flex flex-col items-center justify-center">

            {/* Steps Progress */}
            <div className="w-full max-w-4xl mb-12 flex justify-between items-center text-xs sm:text-sm uppercase tracking-widest font-medium">
                <div className={`flex items-center gap-2 ${step === 'start' ? 'text-cyan-400' : 'text-slate-600'}`}>
                    <div className={`w-3 h-3 rounded-full ${step === 'start' ? 'bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.5)]' : 'bg-slate-800'}`}></div>
                    Start
                </div>
                <div className="h-px bg-slate-800 flex-1 mx-4"></div>
                <div className={`flex items-center gap-2 ${step === 'image' || step === 'analyzing_image' ? 'text-cyan-400' : 'text-slate-600'}`}>
                    <div className={`w-3 h-3 rounded-full ${step === 'image' || step === 'analyzing_image' ? 'bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.5)]' : 'bg-slate-800'}`}></div>
                    Motor
                </div>
                <div className="h-px bg-slate-800 flex-1 mx-4"></div>
                <div className={`flex items-center gap-2 ${step === 'voice' || step === 'analyzing_voice' ? 'text-cyan-400' : 'text-slate-600'}`}>
                    <div className={`w-3 h-3 rounded-full ${step === 'voice' || step === 'analyzing_voice' ? 'bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.5)]' : 'bg-slate-800'}`}></div>
                    Vocal
                </div>
                <div className="h-px bg-slate-800 flex-1 mx-4"></div>
                <div className={`flex items-center gap-2 ${step === 'result' ? 'text-cyan-400' : 'text-slate-600'}`}>
                    <div className={`w-3 h-3 rounded-full ${step === 'result' ? 'bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.5)]' : 'bg-slate-800'}`}></div>
                    Report
                </div>
            </div>

            {/* ERROR MESSAGE */}
            {error && (
                <div className="mb-6 bg-red-500/10 border border-red-500/50 text-red-500 p-4 rounded-lg flex items-center gap-3 backdrop-blur-md">
                    <AlertCircle /> {error}
                </div>
            )}


            {/* STEP 0: START */}
            {step === 'start' && (
                <div className="max-w-xl w-full text-center space-y-8 animate-scale-in">
                    <div className="relative w-32 h-32 mx-auto">
                        <div className="absolute inset-0 bg-cyan-500/20 rounded-full blur-xl animate-pulse-slow"></div>
                        <div className="relative w-full h-full bg-slate-900 border border-slate-700/50 rounded-full flex items-center justify-center shadow-xl">
                            <Brain className="w-16 h-16 text-cyan-400" />
                        </div>
                    </div>

                    <div>
                        <h1 className="text-4xl sm:text-5xl font-bold text-white mb-4 tracking-tight">Diagnostic Procedure</h1>
                        <p className="text-slate-400 text-lg leading-relaxed max-w-md mx-auto">
                            Initiate the multi-modal assessment protocol.
                            Analysis includes <strong>spiral graphometry</strong> and <strong>vocal biomarker</strong> extraction.
                        </p>
                    </div>

                    <Button onClick={handleStart} className="w-full py-4 text-lg font-semibold tracking-wide shadow-cyan-900/20">
                        INITIALIZE SYSTEM <ArrowRight className="ml-2 w-5 h-5" />
                    </Button>
                </div>
            )}


            {/* STEP 1: IMAGE */}
            {step === 'image' && (
                <Card className="max-w-2xl w-full p-8 animate-fade-in-right border-t border-cyan-500/20">
                    <div className="flex justify-between items-start mb-6">
                        <div>
                            <h2 className="text-2xl font-bold text-white">Motor Skill Analysis</h2>
                            <p className="text-slate-400 mt-1">Upload a drawing of a spiral or wave pattern.</p>
                        </div>
                        <div className="bg-cyan-500/10 p-3 rounded-lg border border-cyan-500/20">
                            <Activity className="w-6 h-6 text-cyan-400" />
                        </div>
                    </div>

                    <div className="border border-dashed border-slate-700 bg-slate-900/50 rounded-2xl p-1 text-center hover:border-cyan-500/50 transition-all cursor-pointer relative group overflow-hidden">
                        <input
                            type="file"
                            accept="image/*"
                            onChange={handleImageChange}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                        />

                        <div className="rounded-xl overflow-hidden relative min-h-[300px] flex items-center justify-center bg-slate-950/50">
                            {imagePreview ? (
                                <>
                                    <img src={imagePreview} alt="Preview" className="max-h-[400px] w-auto mx-auto object-contain" />
                                    {/* Scanning Overlay */}
                                    <div className="absolute inset-0 bg-gradient-to-b from-transparent via-cyan-500/10 to-transparent animate-scan pointer-events-none border-b border-cyan-400/50 h-full"></div>
                                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-sm z-10">
                                        <span className="text-white font-medium bg-black/50 px-4 py-2 rounded-full border border-white/10">Click to change</span>
                                    </div>
                                </>
                            ) : (
                                <div className="py-12 px-4">
                                    <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300 border border-slate-800 group-hover:border-cyan-500/50">
                                        <Upload className="w-8 h-8 text-slate-400 group-hover:text-cyan-400 transition-colors" />
                                    </div>
                                    <h3 className="text-lg font-medium text-white mb-2">Drag & Drop or Click to Upload</h3>
                                    <p className="text-sm text-slate-500 max-w-xs mx-auto">
                                        Supports scan or photo. Ensure high contrast.
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="mt-8 flex justify-end">
                        <Button
                            onClick={analyzeImage}
                            disabled={!imageFile}
                            className="w-full sm:w-auto"
                        >
                            Analyze Pattern <ArrowRight className="ml-2 w-4 h-4" />
                        </Button>
                    </div>
                </Card>
            )}

            {/* ANALYZING STATE */}
            {(step === 'analyzing_image' || step === 'analyzing_voice') && (
                <div className="text-center animate-fade-in max-w-md mx-auto">
                    <div className="relative w-40 h-40 mx-auto mb-10">
                        {/* High-tech spinner */}
                        <div className="absolute inset-0 border-4 border-slate-800 rounded-full"></div>
                        <div className="absolute inset-0 border-t-4 border-cyan-500 rounded-full animate-spin"></div>
                        <div className="absolute inset-4 border-4 border-slate-800 rounded-full"></div>
                        <div className="absolute inset-4 border-b-4 border-blue-500 rounded-full animate-spin reverse-spin duration-1000"></div>

                        <div className="absolute inset-0 flex items-center justify-center">
                            <span className="text-2xl font-bold text-white animate-pulse">
                                {Math.floor(Math.random() * 99)}%
                            </span>
                        </div>
                    </div>

                    <h2 className="text-2xl font-bold text-white mb-2 tracking-wide">
                        {step === 'analyzing_image' ? 'PROCESSING IMAGE DATA' : 'ANALYZING AUDIO SPECTRUM'}
                    </h2>
                    <p className="text-cyan-400/80 uppercase text-xs tracking-[0.2em] animate-pulse">
                        Extracting Biomarkers • Calculating Confidence
                    </p>
                </div>
            )}


            {/* STEP 2: VOICE */}
            {step === 'voice' && (
                <Card className="max-w-2xl w-full p-8 animate-fade-in-right border-t border-purple-500/20">
                    <div className="flex justify-between items-start mb-6">
                        <div>
                            <h2 className="text-2xl font-bold text-white">Vocal Biomarker Analysis</h2>
                            <p className="text-slate-400 mt-1">Upload a voice recording (e.g. sustained phonation "aaaaah").</p>
                        </div>
                        <div className="bg-purple-500/10 p-3 rounded-lg border border-purple-500/20">
                            <Mic className="w-6 h-6 text-purple-400" />
                        </div>
                    </div>

                    <div className="border border-dashed border-slate-700 bg-slate-900/50 rounded-2xl p-10 text-center hover:border-purple-500/50 transition-all cursor-pointer relative group">
                        <input
                            type="file"
                            accept="audio/*"
                            onChange={handleVoiceChange}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                        />

                        <div className="py-8">
                            <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300 border border-slate-800 group-hover:border-purple-500/50">
                                <Mic className="w-8 h-8 text-slate-400 group-hover:text-purple-400 transition-colors" />
                            </div>

                            {voiceFile ? (
                                <div className="space-y-4">
                                    <h3 className="text-lg font-bold text-purple-400">{voiceFile.name}</h3>
                                    {/* Simulated waveform visualization */}
                                    <div className="flex items-center justify-center gap-1 h-12">
                                        {[...Array(20)].map((_, i) => (
                                            <div
                                                key={i}
                                                className="w-1 bg-purple-500/50 rounded-full animate-pulse"
                                                style={{
                                                    height: `${30 + Math.random() * 70}%`,
                                                    animationDelay: `${i * 0.05}s`
                                                }}
                                            ></div>
                                        ))}
                                    </div>
                                    <p className="text-xs text-slate-500 uppercase tracking-widest">Ready to Process</p>
                                </div>
                            ) : (
                                <div>
                                    <h3 className="text-lg font-medium text-white mb-2">Click or Drag to Upload Audio</h3>
                                    <p className="text-sm text-slate-500">Supports WAV, MP3 formats</p>
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="mt-8 flex justify-end">
                        <Button
                            onClick={analyzeVoice}
                            disabled={!voiceFile}
                            className="w-full sm:w-auto"
                        >
                            Finalize Analysis <ArrowRight className="ml-2 w-4 h-4" />
                        </Button>
                    </div>
                </Card>
            )}


            {/* STEP 3: RESULT */}
            {step === 'result' && (
                <Card className="max-w-2xl w-full p-0 overflow-hidden animate-scale-in border-t-4 border-t-cyan-500">
                    <div className="bg-slate-900/50 p-8 text-center border-b border-slate-800">
                        <h2 className="text-3xl font-bold text-white mb-2">Diagnostic Report</h2>
                        <p className="text-slate-400 text-sm">{new Date().toLocaleDateString()} • {new Date().toLocaleTimeString()}</p>
                    </div>

                    <div className="p-8">
                        <div className="grid grid-cols-2 gap-6 mb-8">
                            {/* Image Result */}
                            <div className="bg-slate-950 p-6 rounded-xl border border-slate-800 relative overflow-hidden">
                                {imageResult.prediction === 'Healthy'
                                    ? <div className="absolute top-0 right-0 p-2"><CheckCircle className="text-green-500 w-5 h-5" /></div>
                                    : <div className="absolute top-0 right-0 p-2"><AlertCircle className="text-red-500 w-5 h-5" /></div>
                                }
                                <h3 className="text-slate-500 text-xs uppercase tracking-wider mb-2">Motor Skills</h3>
                                <div className={`text-xl font-bold mb-1 ${imageResult.prediction === 'Healthy' ? 'text-green-400' : 'text-red-400'}`}>
                                    {imageResult.prediction}
                                </div>
                                <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2">
                                    <div className={`h-full rounded-full ${imageResult.prediction === 'Healthy' ? 'bg-green-500' : 'bg-red-500'}`} style={{ width: `${imageResult.confidence * 100}%` }}></div>
                                </div>
                            </div>

                            {/* Voice Result */}
                            <div className="bg-slate-950 p-6 rounded-xl border border-slate-800 relative overflow-hidden">
                                {voiceResult.prediction === 'Healthy'
                                    ? <div className="absolute top-0 right-0 p-2"><CheckCircle className="text-green-500 w-5 h-5" /></div>
                                    : <div className="absolute top-0 right-0 p-2"><AlertCircle className="text-red-500 w-5 h-5" /></div>
                                }
                                <h3 className="text-slate-500 text-xs uppercase tracking-wider mb-2">Vocal Pattern</h3>
                                <div className={`text-xl font-bold mb-1 ${voiceResult.prediction === 'Healthy' ? 'text-green-400' : 'text-red-400'}`}>
                                    {voiceResult.prediction}
                                </div>
                                <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2">
                                    <div className={`h-full rounded-full ${voiceResult.prediction === 'Healthy' ? 'bg-green-500' : 'bg-red-500'}`} style={{ width: `${voiceResult.confidence * 100}%` }}></div>
                                </div>
                            </div>
                        </div>

                        {/* Final Verdict */}
                        <div className="text-center bg-gradient-to-br from-slate-900 to-slate-950 rounded-2xl p-8 mb-8 border border-slate-800 shadow-inner">
                            <p className="text-slate-400 mb-2 uppercase text-xs tracking-widest">Aggregate Assessment</p>

                            {(imageResult.prediction === 'Healthy' && voiceResult.prediction === 'Healthy') ? (
                                <>
                                    <h3 className="text-5xl font-bold text-green-400 mb-4 tracking-tight">Negative</h3>
                                    <p className="text-slate-400 max-w-sm mx-auto">
                                        No significant biomarkers detected. Patient is within healthy ranges for both tests.
                                    </p>
                                </>
                            ) : (
                                <>
                                    <h3 className="text-5xl font-bold text-red-500 mb-4 tracking-tight">Positive</h3>
                                    <p className="text-slate-400 max-w-sm mx-auto">
                                        Anomalies detected in
                                        {imageResult.prediction !== 'Healthy' && voiceResult.prediction !== 'Healthy' ? ' both ' :
                                            imageResult.prediction !== 'Healthy' ? ' motor ' : ' vocal '}
                                        parameters. Clinical correlation recommended.
                                    </p>
                                </>
                            )}

                            {/* Combined Confidence (Average) */}
                            <div className="mt-8 pt-6 border-t border-slate-800">
                                <div className="flex justify-between items-end max-w-xs mx-auto">
                                    <span className="text-sm text-slate-500">System Confidence</span>
                                    <span className="text-3xl font-bold text-white">
                                        {(((imageResult.confidence + voiceResult.confidence) / 2) * 100).toFixed(1)}<span className="text-base text-slate-500 font-normal">%</span>
                                    </span>
                                </div>
                            </div>
                        </div>

                        <Button onClick={restart} variant="outline" className="w-full py-4 border-slate-700 hover:bg-slate-800 text-slate-400">
                            Initialize New Procedure
                        </Button>
                    </div>
                </Card>
            )}

        </div>
    );
};

export default Procedure;
