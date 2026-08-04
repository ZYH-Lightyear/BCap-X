import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom/client'

import { App, EMPTY, waitForPaint } from './App'
import { ContextApp } from './context/ContextApp'
import type { VawSnapshot } from './types'
import './styles.css'
import './context/context.css'

function Root() {
  const [snapshot, setSnapshot] = useState<VawSnapshot>(EMPTY)

  useEffect(() => {
    window.__VAW_RENDER__ = async (next) => {
      if (next.schemaVersion === 1) {
        if (next.viewport.width !== 1024 || next.viewport.height !== 576) {
          throw new Error('Unsupported legacy VAW viewport')
        }
      } else if (
        next.schemaVersion !== 3
        || next.schema !== 'vaw-context-v2'
        || next.viewport.width !== 1440
        || next.viewport.height !== 1080
      ) {
        throw new Error('Unsupported VAW Context snapshot')
      }
      document.documentElement.dataset.vawSchema = String(next.schemaVersion)
      setSnapshot(next)
      await waitForPaint()
      document.documentElement.dataset.renderId = next.renderId
    }
    document.documentElement.dataset.vawReady = 'true'
    return () => {
      delete window.__VAW_RENDER__
    }
  }, [])

  return snapshot.schemaVersion === 3
    ? <ContextApp snapshot={snapshot} />
    : <App snapshot={snapshot} />
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
)
