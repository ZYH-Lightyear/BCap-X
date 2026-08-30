import React from 'react'
import ReactDOM from 'react-dom/client'

import { ObservatoryApp } from './observatory/ObservatoryApp'
import './observatory/observatory.css'

ReactDOM.createRoot(document.getElementById('observatory-root')!).render(
  <React.StrictMode>
    <ObservatoryApp />
  </React.StrictMode>,
)
