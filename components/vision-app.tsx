'use client'

import { useEffect, useRef, useState } from 'react'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import * as tf from '@tensorflow/tfjs'
import { khmerLabels } from '@/constants/khmer-labels'

const UI_TEXT = {
  initializing: 'កំពុងដំណើរការ...',
  startingCamera: 'កំពុងបើកកាមេរ៉ា...',
  scanning: 'កំពុងស្កេន...',
  loadingModel: 'កំពុងទាញយកទិន្នន័យ...',
  cameraError: 'មិនអាចប្រើកាមេរ៉ា',
  targetIdentified: 'សម្គាល់ឃើញ',
  nothingDetected: 'មិនមានវត្តមាន',
  version: 'v1.0',
  subtitle: 'ការសម្គាល់វត្ថុ',
}

export default function VisionApp() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null)
  const modelPromiseRef = useRef<Promise<cocoSsd.ObjectDetection> | null>(null)
  const detectFrameRef = useRef<number | null>(null)
  const khmerVoiceRef = useRef<SpeechSynthesisVoice | null>(null)
  const lastSpokenRef = useRef<string>('')
  const speechUnlockedRef = useRef(false)
  const [label, setLabel] = useState(UI_TEXT.startingCamera)
  const [confidence, setConfidence] = useState(0)
  const [isReady, setIsReady] = useState(false)
  const [isCameraReady, setIsCameraReady] = useState(false)
  const [isDetecting, setIsDetecting] = useState(false)
  const [loadingText, setLoadingText] = useState(UI_TEXT.startingCamera)

  async function startCamera(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' }
    })
    streamRef.current = stream
    const video = videoRef.current
    if (!video) return
    video.srcObject = stream

    await new Promise<void>((resolve) => {
      video.onloadedmetadata = () => resolve()
    })
    await video.play().catch(() => undefined)
    setIsCameraReady(true)
  }

  function cacheKhmerVoice() {
    const voices = window.speechSynthesis.getVoices()
    khmerVoiceRef.current = voices.find(v => v.lang === 'km-KH' || v.lang.startsWith('km')) ?? null
  }

  async function loadModel(): Promise<cocoSsd.ObjectDetection> {
    if (modelRef.current) return modelRef.current
    if (modelPromiseRef.current) return modelPromiseRef.current

    modelPromiseRef.current = (async () => {
      await tf.ready()
      const model = await cocoSsd.load({ base: 'lite_mobilenet_v2' })

      // Warm up once to avoid the first detection hitch.
      const warmup = tf.zeros([300, 300, 3], 'int32')
      await model.detect(warmup as tf.Tensor3D, 1, 0.5)
      warmup.dispose()

      modelRef.current = model
      return model
    })()

    return modelPromiseRef.current
  }

  function unlockSpeech() {
    if (speechUnlockedRef.current) return
    speechUnlockedRef.current = true
    const silent = new SpeechSynthesisUtterance('')
    silent.volume = 0
    window.speechSynthesis.speak(silent)
  }

  function speak(text: string) {
    if (text === lastSpokenRef.current) return
    lastSpokenRef.current = text

    const utterance = new SpeechSynthesisUtterance(text)
    if (khmerVoiceRef.current) utterance.voice = khmerVoiceRef.current
    utterance.lang = 'km-KH'
    utterance.rate = 0.9
    utterance.pitch = 1.0
    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(utterance)
  }

  async function detectLoop(model: cocoSsd.ObjectDetection) {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Match canvas internal resolution to its CSS display size
    const displayW = canvas.clientWidth
    const displayH = canvas.clientHeight
    canvas.width = displayW
    canvas.height = displayH

    // Compute object-cover scale & offset to align boxes with the displayed video
    const scale = Math.max(displayW / video.videoWidth, displayH / video.videoHeight)
    const offsetX = (displayW - video.videoWidth * scale) / 2
    const offsetY = (displayH - video.videoHeight * scale) / 2

    const predictions = await model.detect(video, 8, 0.45)
    ctx.clearRect(0, 0, displayW, displayH)

    if (predictions.length > 0) {
      setIsDetecting(true)
      const best = predictions.reduce((a, b) => a.score > b.score ? a : b)

      predictions.forEach(pred => {
        const [bx, by, bw, bh] = pred.bbox
        const x = bx * scale + offsetX
        const y = by * scale + offsetY
        const w = bw * scale
        const h = bh * scale
        const isTop = pred === best
        const color = isTop ? '#22d3ee' : 'rgba(255,255,255,0.5)'

        // glow box
        ctx.shadowColor = color
        ctx.shadowBlur = isTop ? 14 : 5
        ctx.strokeStyle = color
        ctx.lineWidth = isTop ? 2.5 : 1.5
        ctx.strokeRect(x, y, w, h)
        ctx.shadowBlur = 0

        // corner accents
        ctx.strokeStyle = color
        ctx.lineWidth = 3
        ctx.beginPath()
        ctx.moveTo(x, y + 16); ctx.lineTo(x, y); ctx.lineTo(x + 16, y)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(x + w - 16, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - 16)
        ctx.stroke()

        // pill label
        const khmerText = khmerLabels[pred.class] || pred.class
        const labelText = `${khmerText}  ${(pred.score * 100).toFixed(0)}%`
        ctx.font = `bold ${isTop ? 14 : 12}px 'Noto Sans Khmer', sans-serif`
        const tw = ctx.measureText(labelText).width
        const px = x
        const py = y > 30 ? y - 26 : y + h + 4
        const pw = tw + 14
        const ph = 22
        const r = 5

        ctx.fillStyle = color
        ctx.beginPath()
        ctx.moveTo(px + r, py)
        ctx.lineTo(px + pw - r, py)
        ctx.quadraticCurveTo(px + pw, py, px + pw, py + r)
        ctx.lineTo(px + pw, py + ph - r)
        ctx.quadraticCurveTo(px + pw, py + ph, px + pw - r, py + ph)
        ctx.lineTo(px + r, py + ph)
        ctx.quadraticCurveTo(px, py + ph, px, py + ph - r)
        ctx.lineTo(px, py + r)
        ctx.quadraticCurveTo(px, py, px + r, py)
        ctx.closePath()
        ctx.fill()

        ctx.fillStyle = '#fff'
        ctx.fillText(labelText, px + 7, py + 15)
      })

      const khmer = khmerLabels[best.class] || best.class
      setLabel(khmer)
      setConfidence(Math.round(best.score * 100))
      speak(khmerLabels[best.class] || best.class)

    } else {
      setIsDetecting(false)
      setLabel(UI_TEXT.scanning)
      setConfidence(0)
    }

    detectFrameRef.current = requestAnimationFrame(() => {
      void detectLoop(model)
    })
  }

  useEffect(() => {
    let disposed = false

    cacheKhmerVoice()
    window.speechSynthesis.onvoiceschanged = cacheKhmerVoice

    const initialize = async () => {
      try {
        setLoadingText(UI_TEXT.startingCamera)
        const cameraPromise = startCamera()
        const modelPromise = loadModel()
        await cameraPromise
        if (!disposed) {
          setLoadingText(UI_TEXT.loadingModel)
        }
        const model = await modelPromise
        if (disposed) return
        setIsReady(true)
        setLabel(UI_TEXT.scanning)
        void detectLoop(model)
      } catch {
        if (disposed) return
        setLabel(UI_TEXT.cameraError)
        setLoadingText(UI_TEXT.cameraError)
      }
    }

    void initialize()

    return () => {
      disposed = true
      if (detectFrameRef.current !== null) {
        cancelAnimationFrame(detectFrameRef.current)
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
      window.speechSynthesis.cancel()
      window.speechSynthesis.onvoiceschanged = null
    }
  // detectLoop intentionally uses refs/state setters and is started once on mount.
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="relative min-h-dvh overflow-hidden overscroll-none bg-black" onClick={unlockSpeech}>

      {/* Camera — true full screen */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
      />

      {/* Top gradient only — bottom panel has its own background */}
      <div className="absolute top-0 left-0 right-0 h-32 pointer-events-none"
        style={{ background: 'linear-gradient(to bottom, rgba(0,0,0,0.7), transparent)' }} />

      {/* Full loader only before camera is visible */}
      {!isCameraReady && (
        <div className="absolute inset-0 bg-black/90 flex flex-col items-center justify-center gap-5 z-20">
          <p className="font-bold text-white text-2xl tracking-[6px] font-mono">VISIONKH</p>
          <div className="w-10 h-10 border-2 border-stone-700 border-t-cyan-400 rounded-full animate-spin" />
          <p className="text-stone-400 text-sm" style={{ fontFamily: 'var(--font-khmer)' }}>
            {loadingText}
          </p>
        </div>
      )}

      {/* Progressive status while AI model finishes loading */}
      {isCameraReady && !isReady && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
          <div className="px-3 py-1.5 rounded-full border border-cyan-400/40 bg-black/45 backdrop-blur-sm flex items-center gap-2">
            <div className="w-3 h-3 border-2 border-cyan-200/40 border-t-cyan-300 rounded-full animate-spin" />
            <p className="text-cyan-100 text-xs tracking-wide" style={{ fontFamily: 'var(--font-khmer)' }}>
              {loadingText}
            </p>
          </div>
        </div>
      )}

      {/* Top bar — floats over camera */}
      <div
        className="absolute top-0 left-0 right-0 z-10 flex justify-between items-center px-5 py-3"
        style={{ paddingTop: 'max(0.75rem, env(safe-area-inset-top))' }}
      >
        <div>
          <p className="font-bold text-white text-lg tracking-[4px] font-mono">VISIONKH</p>
          <p className="text-white/40 text-base font-mono tracking-widest">{UI_TEXT.version} · {UI_TEXT.subtitle}</p>
        </div>
      </div>

      {/* Bottom panel — floats over camera */}
      <div
        className="absolute bottom-0 left-0 right-0 z-10 px-5 pt-3"
        style={{
          paddingBottom: 'max(1rem, env(safe-area-inset-bottom))',
          background: 'linear-gradient(to top, rgba(0,0,0,0.8) 70%, transparent)',
        }}
      >
        <div className="flex items-end justify-between gap-4">

          {/* Label */}
          <div className="flex-1 min-w-0">
            <p className="text-white/40 text-base mb-1" style={{ fontFamily: 'var(--font-khmer)' }}>
              {isDetecting ? UI_TEXT.targetIdentified : UI_TEXT.nothingDetected}
            </p>
            <p
              className="font-bold truncate transition-all duration-300 pb-2"
              style={{
                fontFamily: 'var(--font-khmer)',
                fontSize: 'clamp(1.75rem, 7vw, 2.75rem)',
                color: isDetecting ? '#22d3ee' : 'rgba(255,255,255,0.25)',
                textShadow: isDetecting ? '0 0 30px rgba(34,211,238,0.4)' : 'none',
              }}
            >
              {label}
            </p>
          </div>

          {/* Confidence ring */}
          <div
            className="shrink-0 w-18 h-18 rounded-full flex flex-col items-center justify-center border-2 backdrop-blur-sm transition-all duration-500"
            style={{
              width: '72px',
              height: '72px',
              borderColor: isDetecting ? '#22d3ee' : 'rgba(255,255,255,0.15)',
              background: isDetecting ? 'rgba(34,211,238,0.1)' : 'rgba(0,0,0,0.3)',
              boxShadow: isDetecting ? '0 0 20px rgba(34,211,238,0.2)' : 'none'
            }}
          >
            <p className="font-bold text-xl leading-none font-mono"
              style={{ color: isDetecting ? '#22d3ee' : 'rgba(255,255,255,0.25)' }}>
              {confidence > 0 ? confidence : '--'}
            </p>
            {confidence > 0 && (
              <p className="text-[9px] font-mono" style={{ color: '#22d3ee' }}>%</p>
            )}
          </div>

        </div>

      </div>

    </div>
  )
}
