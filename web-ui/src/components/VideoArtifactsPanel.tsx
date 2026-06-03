import { useEffect, useMemo, useState } from 'react';
import type { ArtifactItem, ArtifactListResponse, SessionState } from '../types/messages';

interface VideoArtifactsPanelProps {
  sessionId: string | null;
  trialState: SessionState;
}

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function VideoArtifactsPanel({ sessionId, trialState }: VideoArtifactsPanelProps) {
  const [artifacts, setArtifacts] = useState<ArtifactListResponse | null>(null);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isLive = trialState === 'running' || trialState === 'awaiting_user_input';

  useEffect(() => {
    setArtifacts(null);
    setSelectedPath(null);
    setError(null);
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) return;

    let cancelled = false;

    async function poll() {
      while (!cancelled) {
        try {
          const resp = await fetch(`/api/artifacts/${sessionId}`);
          if (!resp.ok) throw new Error(await resp.text());
          const data = await resp.json() as ArtifactListResponse;
          if (cancelled) return;
          setArtifacts(data);
          setError(null);

          const latest = data.videos[0];
          setSelectedPath((current) => {
            if (!latest) return current;
            const stillExists = current && data.videos.some((video) => video.path === current);
            return stillExists ? current : latest.path;
          });
        } catch (err) {
          if (!cancelled) {
            setError(err instanceof Error ? err.message : 'Failed to load artifacts');
          }
        }
        await new Promise((resolve) => setTimeout(resolve, isLive ? 2500 : 8000));
      }
    }

    poll();
    return () => { cancelled = true; };
  }, [sessionId, isLive]);

  const selectedVideo = useMemo<ArtifactItem | null>(() => {
    if (!artifacts?.videos.length) return null;
    return artifacts.videos.find((video) => video.path === selectedPath) || artifacts.videos[0];
  }, [artifacts, selectedPath]);

  return (
    <div className="h-full flex flex-col border-t border-surface-border bg-surface-raised min-h-0">
      <div className="flex-shrink-0 px-5 py-3 bg-surface-raised border-b border-surface-border flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-6 h-6 rounded-md bg-accent/10 border border-accent/20 flex items-center justify-center">
            <svg className="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.55-2.28A1 1 0 0121 8.62v6.76a1 1 0 01-1.45.9L15 14M5 6h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2z" />
            </svg>
          </div>
          <div className="min-w-0">
            <div className="text-sm font-bold font-display tracking-wide uppercase text-text-primary">Rendered Video</div>
            <div className="text-xs text-text-tertiary truncate">
              {artifacts?.output_dir || 'Waiting for output directory'}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-text-tertiary flex-shrink-0">
          {isLive && <span className="w-2 h-2 rounded-full bg-accent animate-pulse" />}
          <span>{artifacts?.videos.length || 0} videos</span>
        </div>
      </div>

      <div className="flex-1 min-h-0 flex flex-col p-3 gap-3">
        {selectedVideo ? (
          <>
            <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-surface-border bg-black flex items-center">
              <video
                key={`${selectedVideo.url}:${selectedVideo.mtime}`}
                src={selectedVideo.url}
                controls
                muted
                playsInline
                className="w-full h-full object-contain"
              />
            </div>
            <div className="flex-shrink-0 flex items-center gap-2">
              <select
                value={selectedVideo.path}
                onChange={(e) => setSelectedPath(e.target.value)}
                className="flex-1 min-w-0 px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-xs text-text-primary font-mono focus:outline-none focus:ring-1 focus:ring-accent/40 focus:border-accent/40"
              >
                {artifacts?.videos.map((video) => (
                  <option key={video.path} value={video.path}>
                    {video.path} · {formatSize(video.size)}
                  </option>
                ))}
              </select>
              <a
                href={selectedVideo.url}
                target="_blank"
                rel="noreferrer"
                className="px-3 py-2 text-xs font-display bg-surface-overlay text-text-secondary rounded-md hover:text-accent hover:bg-surface-border transition-colors"
              >
                Open
              </a>
            </div>
            <div className="flex-shrink-0 text-xs text-text-tertiary flex items-center justify-between">
              <span>{selectedVideo.path}</span>
              <span>{artifacts?.overlays.length || 0} overlays · {artifacts?.line_traces.length || 0} traces</span>
            </div>
          </>
        ) : (
          <div className="flex-1 rounded-lg border border-dashed border-surface-border bg-surface-sunken flex flex-col items-center justify-center text-center px-6">
            <div className="text-sm font-display text-text-secondary mb-2">
              {sessionId ? 'Waiting for rendered video' : 'Start a trial to show video'}
            </div>
            <div className="text-xs text-text-tertiary max-w-sm">
              每个 code block 执行完后会生成 `video_partial_turn_XX.mp4`，trial 结束后会出现最终视频。
            </div>
            {error && <div className="mt-3 text-xs text-red-400 max-w-sm truncate">{error}</div>}
          </div>
        )}
      </div>
    </div>
  );
}
