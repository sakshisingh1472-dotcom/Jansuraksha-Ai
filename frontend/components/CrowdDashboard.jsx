"use client";
import { useState } from "react";

export default function CrowdDashboard() {
  const [count, setCount] = useState(null);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setPreview(URL.createObjectURL(file));
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch("http://127.0.0.1:8000/analyze-image", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setCount(data.people_count);
      setStatus(data.status);
    } catch (err) {
      alert("Backend 8000 pe nahi chal raha!");
    }
    setLoading(false);
  };

  const isAlert = count!== null && count > 40;

  return (
    <div className="min-h-screen bg-black text-white p-6">
      <h1 className="text-3xl font-bold">JANSURAKSHA AI - Dashboard</h1>
      <p className="text-gray-400 mb-6">SIH26187 | Backend: YOLOv8l + Tiling</p>
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-zinc-900 p-6 rounded-2xl">
          <h2 className="text-xl mb-4">Upload Crowd Image</h2>
          <input type="file" onChange={handleUpload} className="bg-zinc-800 p-3 rounded w-full" />
          {preview && <img src={preview} className="mt-4 rounded-xl max-h-[400px] w-full object-cover" />}
        </div>
        <div className="bg-zinc-900 p-6 rounded-2xl">
          <h2 className="text-xl mb-4">Live Analysis</h2>
          {loading? <p className="text-yellow-400 animate-pulse">Analysing...</p> : count!== null? <>
            <div className="text-6xl font-black">{count}</div>
            <div className={`mt-3 inline-block px-4 py-2 rounded-full font-bold ${isAlert? "bg-red-600 animate-pulse" : "bg-green-600"}`}>
              {status} {isAlert? "🚨" : "✅"}
            </div>
          </> : <p className="text-gray-500">Photo upload kar result yahan aayega</p>}
        </div>
      </div>
    </div>
  );
}