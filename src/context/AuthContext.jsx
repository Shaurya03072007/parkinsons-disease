import React, { createContext, useContext, useState, useEffect } from 'react';
import { jwtDecode } from "jwt-decode";
import { useNavigate } from 'react-router-dom';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check for existing token
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const decoded = jwtDecode(token);
        setUser({
          username: decoded.name || decoded.email,
          email: decoded.email,
          picture: decoded.picture
        });
      } catch (e) {
        localStorage.removeItem('token');
      }
    }
    setLoading(false);
  }, []);

  const googleLogin = async (credential) => {
    try {
      // 1. Decode token client-side for immediate UI feedback
      const decoded = jwtDecode(credential);

      // 2. Store token
      localStorage.setItem('token', credential);

      // 3. Set user state
      setUser({
        username: decoded.name,
        email: decoded.email,
        picture: decoded.picture
      });

      // 4. Verify with backend
      const response = await fetch('http://localhost:8000/auth/google', {
        method: 'POST',
        body: JSON.stringify({ token: credential }),
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error('Backend verification failed');
      }

      return true;
    } catch (error) {
      console.error("Auth Error", error);
      return false;
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('token');
  };

  return (
    <AuthContext.Provider value={{ user, loading, googleLogin, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
