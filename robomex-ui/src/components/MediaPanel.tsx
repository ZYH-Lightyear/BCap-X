import type { MediaItem } from '../types'
import { shortName } from '../status'

type Props = {
  media: MediaItem[]
  title?: string
}

export function MediaPanel({ media, title = 'Media' }: Props) {
  const images = media.filter((m) => m.kind === 'image')
  const videos = media.filter((m) => m.kind === 'video')

  return (
    <div className="flex h-full min-h-0 flex-col rounded border border-ink-200 bg-white">
      <div className="border-b border-ink-200 px-3 py-2 text-xs font-semibold uppercase tracking-wide">
        {title}
      </div>
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        {videos.slice(0, 3).map((item) => (
          <div key={item.path}>
            <video src={item.url} controls muted playsInline className="w-full rounded border border-ink-200 bg-black" />
            <div className="mt-1 truncate font-mono text-[10px] text-ink-500">{shortName(item.path)}</div>
          </div>
        ))}
        {images.slice(0, 8).map((item) => (
          <a key={item.path} href={item.url} target="_blank" rel="noreferrer" className="block">
            <img
              src={item.url}
              alt={item.path}
              className="max-h-48 w-full rounded border border-ink-200 object-contain bg-ink-900"
            />
            <div className="mt-1 truncate font-mono text-[10px] text-ink-500">{shortName(item.path)}</div>
          </a>
        ))}
        {media.length === 0 && (
          <div className="text-xs text-ink-500">No png/mp4 artifacts in scope.</div>
        )}
      </div>
    </div>
  )
}
