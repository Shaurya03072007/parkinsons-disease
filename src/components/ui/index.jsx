import React from 'react';

export const Button = ({ children, className = '', variant = 'primary', ...props }) => {
    const baseStyles = "px-6 py-3 rounded-lg font-medium transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center";

    const variants = {
        primary: "bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/40 focus:ring-cyan-500",
        secondary: "bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 focus:ring-slate-500",
        outline: "bg-transparent border border-slate-600 text-slate-400 hover:border-cyan-500 hover:text-cyan-400 focus:ring-cyan-500",
        ghost: "bg-transparent hover:bg-slate-800 text-slate-400 hover:text-white",
        danger: "bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-500/20 focus:ring-red-500"
    };

    return (
        <button
            className={`${baseStyles} ${variants[variant] || variants.primary} ${className}`}
            {...props}
        >
            {children}
        </button>
    );
};

export const Card = ({ children, className = '', ...props }) => {
    return (
        <div
            className={`bg-slate-900/50 backdrop-blur-md border border-slate-800/50 rounded-2xl p-6 shadow-xl ${className}`}
            {...props}
        >
            {children}
        </div>
    );
};

export const Input = ({ className = '', ...props }) => {
    return (
        <input
            className={`w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-slate-200 placeholder-slate-600 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-colors outline-none ${className}`}
            {...props}
        />
    );
};
