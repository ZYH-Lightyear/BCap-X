import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom/client'

import { ContextApp } from './context/ContextApp'
import type { ContextSnapshot } from './types'
import './context/context.css'

function waitForPaint(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      requestAnimationFrame(async () => {
        await document.fonts.ready
        const pending = Array.from(document.images)
          .filter((image) => !image.complete)
          .map(
            (image) => new Promise<void>((done) => {
              image.addEventListener('load', () => done(), { once: true })
              image.addEventListener('error', () => done(), { once: true })
            }),
          )
        await Promise.all(pending)
        resolve()
      })
    })
  })
}

function Root() {
  const [snapshot, setSnapshot] = useState<ContextSnapshot | null>(null)

  useEffect(() => {
    window.__VAW_RENDER__ = async (next) => {
      if (
        next.schemaVersion !== 18
        || next.schema !== 'vaw-context-v17-contact-semantics'
        || next.viewport.width !== 1920
        || next.viewport.height !== 1080
      ) {
        throw new Error('Unsupported VAW Context snapshot')
      }
      setSnapshot(next)
      await waitForPaint()
      document.documentElement.dataset.renderId = next.renderId
    }
    document.documentElement.dataset.vawReady = 'true'
    return () => {
      delete window.__VAW_RENDER__
    }
  }, [])

  return snapshot ? <ContextApp snapshot={snapshot} /> : null
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
)
