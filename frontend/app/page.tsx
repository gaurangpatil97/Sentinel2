"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type DetectionResponse = {
  output_image: string;
  helmet_count: number;
  no_helmet_count: number;
  bicyclist_count: number;
  violation: boolean;
  driver_count: number;
};

function toImageSrc(image: string) {
  return image.startsWith("data:") ? image : `data:image/png;base64,${image}`;
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreviewSrc(null);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewSrc(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const outputSrc = useMemo(() => {
    if (!result?.output_image) {
      return null;
    }

    return toImageSrc(result.output_image);
  }, [result]);

  const selectFile = (nextFile: File | null) => {
    if (!nextFile) {
      return;
    }

    if (!nextFile.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }

    setError(null);
    setResult(null);
    setFile(nextFile);
  };

  const handleDetect = async () => {
    if (!file) {
      setError("Upload an image before running detection.");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);

    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:5000/detect", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as
          | { detail?: string; error?: string }
          | null;
        throw new Error(payload?.detail ?? payload?.error ?? `Request failed with status ${response.status}`);
      }

      const data = (await response.json()) as DetectionResponse;
      setResult(data);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Detection failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(34,197,94,0.18),transparent_30%),radial-gradient(circle_at_bottom_right,rgba(59,130,246,0.14),transparent_28%),linear-gradient(180deg,#040712_0%,#07111f_55%,#03050a_100%)] px-4 py-8 text-slate-100 sm:px-6 lg:px-8">
      <div className="mx-auto flex min-h-[calc(100vh-4rem)] w-full max-w-6xl items-center">
        <section className="w-full rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl shadow-black/30 backdrop-blur-md md:p-8">
          <div className="mb-8 space-y-3">
            <p className="text-sm font-medium uppercase tracking-[0.3em] text-emerald-300/80">
              Sentinel Vision
            </p>
            <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-5xl">
              Sentinel — Helmet Detection
            </h1>
            <p className="max-w-2xl text-sm leading-6 text-slate-300 sm:text-base">
              Drop an image, preview it locally, and send it to the detection backend for annotated helmet analysis.
            </p>
          </div>

          <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
            <div className="space-y-4">
              <div
                onClick={() => inputRef.current?.click()}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragActive(false);
                  selectFile(event.dataTransfer.files?.[0] ?? null);
                }}
                className={`flex min-h-72 cursor-pointer flex-col items-center justify-center rounded-3xl border border-dashed px-6 text-center transition duration-200 ${
                  dragActive
                    ? "border-emerald-400 bg-emerald-400/10"
                    : "border-white/15 bg-slate-950/45 hover:border-white/30 hover:bg-slate-950/65"
                }`}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(event) => selectFile(event.target.files?.[0] ?? null)}
                />

                <div className="flex max-w-md flex-col items-center gap-3">
                  <div className="flex h-14 w-14 items-center justify-center rounded-full border border-emerald-400/25 bg-emerald-400/10 text-2xl text-emerald-300">
                    +
                  </div>
                  <div>
                    <p className="text-lg font-medium text-white">Drag and drop an image here</p>
                    <p className="mt-1 text-sm text-slate-400">Or click to browse from your device</p>
                  </div>
                </div>
              </div>

              {previewSrc ? (
                <div className="overflow-hidden rounded-3xl border border-white/10 bg-slate-950/60">
                  <div className="border-b border-white/10 px-4 py-3 text-sm font-medium text-slate-300">
                    Preview
                  </div>
                  <div className="bg-black/30 p-4">
                    <img
                      src={previewSrc}
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
                      Sends the selected image to <span className="text-slate-200">http://localhost:5000/detect</span>.
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={handleDetect}
                    disabled={!file || loading}
                    className="inline-flex h-11 items-center justify-center rounded-full bg-emerald-400 px-5 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
                  >
                    {loading ? "Detecting..." : "Detect"}
                  </button>
                </div>

                {loading ? (
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
                      The backend response will render here after detection.
                    </p>
                  </div>
                </div>

                {outputSrc ? (
                  <div className="overflow-hidden rounded-2xl border border-white/10 bg-black/30 p-3">
                    <img
                      src={outputSrc}
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
