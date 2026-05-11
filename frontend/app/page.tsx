"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type DetectionResponse = {
  output_image: string;
  helmet_count: number;
  no_helmet_count: number;
  bicyclist_count: number;
  violation: boolean;
};

function getImageSrc(image: string) {
  if (image.startsWith("data:")) {
    return image;
  }

  return `data:image/jpeg;base64,${image}`;
}

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const outputImageSrc = useMemo(() => {
    if (!result?.output_image) {
      return null;
    }

    return getImageSrc(result.output_image);
  }, [result]);

  const handleFile = (file: File | null) => {
    if (!file) {
      return;
    }

    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file.");
      return;
    }

    setError(null);
    setResult(null);
    setSelectedFile(file);
  };

  const handleDetect = async () => {
    if (!selectedFile) {
      setError("Select an image before running detection.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:5000/detect", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Detection failed with status ${response.status}.`);
      }

      const data = (await response.json()) as DetectionResponse;
      setResult(data);
    } catch (fetchError) {
      setError(
        fetchError instanceof Error
          ? fetchError.message
          : "Detection request failed."
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(34,197,94,0.18),transparent_32%),radial-gradient(circle_at_bottom_right,rgba(59,130,246,0.14),transparent_28%),linear-gradient(180deg,#040712_0%,#07111f_55%,#03050a_100%)] px-4 py-8 text-slate-100 sm:px-6 lg:px-8">
      <div className="mx-auto flex min-h-[calc(100vh-4rem)] w-full max-w-6xl flex-col justify-center gap-8">
        <section className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl shadow-black/30 backdrop-blur md:p-8">
          <div className="mb-8 flex flex-col gap-3">
            <p className="text-sm font-medium uppercase tracking-[0.32em] text-emerald-300/80">
              Sentinel Vision
            </p>
            <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-5xl">
              Sentinel — Helmet Detection
            </h1>
            <p className="max-w-2xl text-sm leading-6 text-slate-300 sm:text-base">
              Upload an image, run inference against the local detection backend,
              and review the annotated result and violation summary.
            </p>
          </div>

          <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
            <div className="space-y-4">
              <div
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setIsDragging(false);
                  handleFile(event.dataTransfer.files?.[0] ?? null);
                }}
                onClick={() => fileInputRef.current?.click()}
                className={`flex min-h-72 cursor-pointer flex-col items-center justify-center rounded-3xl border border-dashed px-6 text-center transition duration-200 ${
                  isDragging
                    ? "border-emerald-400 bg-emerald-400/10"
                    : "border-white/15 bg-slate-950/45 hover:border-white/30 hover:bg-slate-950/65"
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(event) => handleFile(event.target.files?.[0] ?? null)}
                />

                <div className="flex max-w-md flex-col items-center gap-3">
                  <div className="flex h-14 w-14 items-center justify-center rounded-full border border-emerald-400/25 bg-emerald-400/10 text-2xl text-emerald-300">
                    +
                  </div>
                  <div>
                    <p className="text-lg font-medium text-white">
                      Drag and drop an image here
                    </p>
                    <p className="mt-1 text-sm text-slate-400">
                      Or click to browse from your device
                    </p>
                  </div>
                </div>
              </div>

              {previewUrl ? (
                <div className="overflow-hidden rounded-3xl border border-white/10 bg-slate-950/60">
                  <div className="border-b border-white/10 px-4 py-3 text-sm font-medium text-slate-300">
                    Preview
                  </div>
                  <div className="bg-black/30 p-4">
                    <img
                      src={previewUrl}
                      alt="Uploaded preview"
                      className="max-h-128 w-full rounded-2xl object-contain"
                    />
                  </div>
                </div>
              ) : (
                <div className="rounded-3xl border border-white/10 bg-slate-950/40 p-5 text-sm text-slate-400">
                  Selected image preview will appear here before detection.
                </div>
              )}
            </div>

            <div className="space-y-4">
              <div className="rounded-3xl border border-white/10 bg-slate-950/50 p-5">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h2 className="text-lg font-semibold text-white">Detection</h2>
                    <p className="text-sm text-slate-400">
                      Sends the uploaded image to the backend as multipart/form-data.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleDetect}
                    disabled={!selectedFile || isLoading}
                    className="inline-flex h-11 items-center justify-center rounded-full bg-emerald-400 px-5 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
                  >
                    {isLoading ? "Detecting..." : "Detect"}
                  </button>
                </div>

                {isLoading ? (
                  <div className="mt-6 flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-300">
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-400 border-t-emerald-300" />
                    Processing image...
                  </div>
                ) : null}

                {error ? (
                  <div className="mt-6 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                    {error}
                  </div>
                ) : null}
              </div>

              <div className="rounded-3xl border border-white/10 bg-slate-950/50 p-5">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div>
                    <h2 className="text-lg font-semibold text-white">Annotated Output</h2>
                    <p className="text-sm text-slate-400">
                      The backend response appears here after detection completes.
                    </p>
                  </div>
                </div>

                {outputImageSrc ? (
                  <div className="overflow-hidden rounded-2xl border border-white/10 bg-black/30 p-3">
                    <img
                      src={outputImageSrc}
                      alt="Annotated detection output"
                      className="max-h-104 w-full rounded-xl object-contain"
                    />
                  </div>
                ) : (
                  <div className="flex min-h-48 items-center justify-center rounded-2xl border border-dashed border-white/10 bg-black/20 px-6 text-center text-sm text-slate-500">
                    Run detection to view the annotated image.
                  </div>
                )}

                {result ? (
                  <div className="mt-4 grid gap-3 sm:grid-cols-3">
                    <StatCard label="Helmets detected" value={result.helmet_count} />
                    <StatCard label="No-helmets detected" value={result.no_helmet_count} />
                    <StatCard label="Bicyclists detected" value={result.bicyclist_count} />
                  </div>
                ) : null}

                {result?.violation ? (
                  <div className="mt-4 rounded-2xl border border-red-500/40 bg-red-500/15 px-4 py-3 text-sm font-semibold text-red-100">
                    Violation alert: one or more no-helmet detections were found.
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-4">
      <p className="text-xs uppercase tracking-[0.22em] text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-white">{value}</p>
    </div>
  );
}
