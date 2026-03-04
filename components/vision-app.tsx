'use client'

import { useEffect, useRef, useState } from 'react'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import '@tensorflow/tfjs'
import { khmerLabels } from '@/constants/khmer-labels'

const UI_TEXT = {
  initializing: 'កំពុងចាប់ផ្តើម...',
  scanning: 'កំពុងស្កេន...',
  systemOnline: 'ប្រព័ន្ធអនឡាញ',
  loadingModel: 'កំពុងផ្ទុកម៉ូដែល...',
  targetIdentified: 'គោលដៅត្រូវបានកំណត់',
  confidence: 'ភាពជឿជាក់',
  nothingDetected: 'រកមិនឃើញ...',
  version: 'v1.0 // ការរកឃើញវត្ថុ',
}

export default function VisionApp() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const lastSpokenRef = useRef<string>('')
  const [label, setLabel] = useState(UI_TEXT.initializing)
  const [confidence, setConfidence] = useState(0)
  const [isReady, setIsReady] = useState(false)

  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' }
    })
    if (videoRef.current) {
      videoRef.current.srcObject = stream
      videoRef.current.onloadeddata = () => loadModelAndDetect()
    }
  }

  async function loadModelAndDetect() {
    const model = await cocoSsd.load()
    setIsReady(true)
    setLabel(UI_TEXT.scanning)
    detectLoop(model)
  }

  function speak(text: string) {
    // Only speak if it's a different object than last time
    if (text === lastSpokenRef.current) return
    lastSpokenRef.current = text

    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = 'km-KH'
    utterance.rate = 0.9
    utterance.pitch = 0.8
    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(utterance)
  }

  async function detectLoop(model: cocoSsd.ObjectDetection) {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const predictions = await model.detect(video)

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (predictions.length > 0) {
      const best = predictions.reduce((a, b) => a.score > b.score ? a : b)

      predictions.forEach(pred => {
        const [x, y, w, h] = pred.bbox
        const isTop = pred === best

        ctx.shadowColor = isTop ? '#00ff88' : '#0088ff'
        ctx.shadowBlur = 12
        ctx.strokeStyle = isTop ? '#00ff88' : '#0088ff'
        ctx.lineWidth = isTop ? 2.5 : 1.5
        ctx.strokeRect(x, y, w, h)

        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.moveTo(x, y + 15)
        ctx.lineTo(x, y)
        ctx.lineTo(x + 15, y)
        ctx.stroke()

        ctx.beginPath()
        ctx.moveTo(x + w - 15, y + h)
        ctx.lineTo(x + w, y + h)
        ctx.lineTo(x + w, y + h - 15)
        ctx.stroke()

        ctx.shadowBlur = 0
        ctx.fillStyle = isTop ? 'rgba(0,255,136,0.15)' : 'rgba(0,136,255,0.15)'
        ctx.fillRect(x, y - 24, w, 24)

        ctx.fillStyle = isTop ? '#00ff88' : '#0088ff'
        ctx.font = 'bold 13px monospace'
        ctx.fillText(
          `${khmerLabels[pred.class] || pred.class.toUpperCase()}  ${(pred.score * 100).toFixed(0)}%`,
          x + 6,
          y - 7
        )
      })

      const khmer = khmerLabels[best.class] || best.class.toUpperCase()
      setLabel(khmer)
      setConfidence(Math.round(best.score * 100))
      speak(khmerLabels[best.class] || best.class)

    } else {
      setLabel(UI_TEXT.scanning)
      setConfidence(0)
    }

    requestAnimationFrame(() => detectLoop(model))
  }

  useEffect(() => {
    startCamera()
  }, [])

  return (
    <div className="relative flex flex-col items-center justify-center min-h-dvh bg-black overflow-hidden overscroll-none">

      <div className="absolute inset-0 opacity-10"
        style={{
          backgroundImage: `linear-gradient(#00ff88 1px, transparent 1px), linear-gradient(90deg, #00ff88 1px, transparent 1px)`,
          backgroundSize: '40px 40px'
        }}
      />

      <div className="absolute top-0 left-0 right-0 flex justify-between items-center px-6 py-4 z-10" style={{ paddingTop: 'max(1rem, env(safe-area-inset-top))' }}>
        <div>
          <p className="text-green-400 font-mono text-xs tracking-widest">VISIONX</p>
          <p className="text-green-900 font-mono text-xs">{UI_TEXT.version}</p>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isReady ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'}`} />
          <p className="text-green-400 font-mono text-xs tracking-widest">
            {isReady ? UI_TEXT.systemOnline : UI_TEXT.loadingModel}
          </p>
        </div>
      </div>

      <div className="relative w-full max-w-2xl">
        <div className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-green-400 z-10" />
        <div className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-green-400 z-10" />
        <div className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-green-400 z-10" />
        <div className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-green-400 z-10" />

        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full rounded-sm opacity-90"
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
        />
      </div>

      <div className="absolute bottom-0 left-0 right-0 px-6 py-6 z-10" style={{ paddingBottom: 'max(1.5rem, env(safe-area-inset-bottom))' }}>
        <div className="flex justify-between items-end">
          <div>
            <p className="text-green-900 font-mono text-xs tracking-widest mb-1">{UI_TEXT.targetIdentified}</p>
            <p className="text-green-400 text-3xl sm:text-5xl font-bold drop-shadow-lg"
              style={{ fontFamily: "'Khmer OS', 'Noto Sans Khmer', sans-serif" }}>
              {label}
            </p>
          </div>
          <div className="text-right">
            <p className="text-green-900 font-mono text-xs tracking-widest mb-1">{UI_TEXT.confidence}</p>
            <p className="text-green-400 font-mono text-2xl sm:text-4xl font-bold">
              {confidence > 0 ? `${confidence}%` : '--'}
            </p>
          </div>
        </div>

        <div className="mt-3 h-1 bg-green-900 rounded-full overflow-hidden">
          <div
            className="h-full bg-green-400 transition-all duration-300"
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>
    </div>
  )
}