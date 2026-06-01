import { existsSync } from 'node:fs'
import path from 'node:path'
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const isWindows = process.platform === 'win32'
const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const presentationDir = path.join(rootDir, 'presentation')
const backendDir = path.join(rootDir, 'fastapi')

const defaults = {
  venvDir: '.venv',
  frontendHost: '0.0.0.0',
  frontendPort: '5173',
  backendHost: '0.0.0.0',
  backendPort: '8000',
  remoteHost: '0.0.0.0',
  remotePort: '4174',
  viteRemoteWsUrl: '',
  python: isWindows ? 'python' : 'python3',
}

const parseArgs = (argv) => {
  const [target = 'help', ...rest] = argv
  const options = { ...defaults }

  for (let index = 0; index < rest.length; index += 1) {
    const arg = rest[index]
    if (!arg.startsWith('--')) continue

    const [rawKey, inlineValue] = arg.slice(2).split('=', 2)
    const key = rawKey.replace(/-([a-z])/g, (_, char) => char.toUpperCase())
    const value = inlineValue ?? rest[index + 1]
    if (inlineValue === undefined) index += 1

    if (key in options && value !== undefined) {
      options[key] = value
    }
  }

  return { target, options }
}

const executable = (name) => {
  return name
}

const needsShell = (command) => {
  return isWindows && command === 'npm'
}

const venvPython = (options) => {
  const pythonPath = isWindows
    ? path.join(options.venvDir, 'Scripts', 'python.exe')
    : path.join(options.venvDir, 'bin', 'python')
  return path.resolve(rootDir, pythonPath)
}

const run = (command, args, spawnOptions = {}) => {
  let child
  try {
    child = spawn(executable(command), args, {
      cwd: rootDir,
      stdio: 'inherit',
      shell: needsShell(command),
      ...spawnOptions,
    })
  } catch (error) {
    throw new Error(`Failed to start ${command}: ${error.message}`)
  }

  return new Promise((resolve, reject) => {
    child.on('error', reject)
    child.on('exit', (code, signal) => {
      if (code === 0) resolve()
      else reject(new Error(`${command} exited with ${signal ?? code}`))
    })
  })
}

const spawnService = (name, command, args, spawnOptions = {}) => {
  let child
  try {
    child = spawn(executable(command), args, {
      cwd: rootDir,
      stdio: 'inherit',
      shell: needsShell(command),
      ...spawnOptions,
    })
  } catch (error) {
    throw new Error(`[${name}] failed to start ${command}: ${error.message}`)
  }

  child.on('error', (error) => {
    console.error(`[${name}] failed to start: ${error.message}`)
  })

  return child
}

const stopChild = (child) => {
  if (child.killed || child.exitCode !== null) return

  if (isWindows && child.pid) {
    spawn('taskkill', ['/pid', String(child.pid), '/t', '/f'], {
      stdio: 'ignore',
      shell: false,
    })
    return
  }

  child.kill('SIGTERM')
}

const ensureBackendDeps = (options) => {
  const python = venvPython(options)
  if (!existsSync(python)) {
    throw new Error(`Missing ${path.relative(rootDir, python)}. Run: make setup VENV_DIR=${options.venvDir}`)
  }
}

const ensureFrontendDeps = () => {
  const nodeModules = path.join(presentationDir, 'node_modules')
  if (!existsSync(nodeModules)) {
    throw new Error('Missing presentation/node_modules. Run: make setup')
  }
}

const printHelp = (options) => {
  console.log(`Presentation app targets:
  make setup       Install backend and frontend dependencies
  make examples    Generate fastapi/examples.json for the live demo
  make backend     Run FastAPI backend on port ${options.backendPort}
  make frontend    Run Vite presentation on port ${options.frontendPort}
  make laser       Run WebSocket laser pointer / phone remote hub on port ${options.remotePort}
  make run         Run backend, frontend, and laser pointer hub together

Open presentation: http://localhost:${options.frontendPort}
Open phone remote: http://<laptop-ip>:${options.frontendPort}/remote

Override the venv folder with: make setup VENV_DIR=.presentation`)
}

const setup = async (options) => {
  await run('uv', ['venv', '--python', options.python, options.venvDir])
  await run('uv', ['pip', 'install', '--python', venvPython(options), '-r', path.join('fastapi', 'requirements-api.txt')])
  await run('npm', ['--prefix', 'presentation', 'install'])
}

const examples = async (options) => {
  ensureBackendDeps(options)
  await run(venvPython(options), [path.join('fastapi', 'generate_examples.py')])
}

const backend = async (options) => {
  ensureBackendDeps(options)
  await run(
    venvPython(options),
    ['-m', 'uvicorn', 'main:app', '--reload', '--host', options.backendHost, '--port', options.backendPort],
    { cwd: backendDir },
  )
}

const frontend = async (options) => {
  ensureFrontendDeps()
  await run('npm', ['--prefix', 'presentation', 'run', 'dev', '--', '--host', options.frontendHost, '--port', options.frontendPort], {
    env: { ...process.env, VITE_REMOTE_WS_URL: options.viteRemoteWsUrl },
  })
}

const remote = async (options) => {
  ensureFrontendDeps()
  await run('npm', ['--prefix', 'presentation', 'run', 'remote'], {
    env: { ...process.env, REMOTE_HOST: options.remoteHost, REMOTE_PORT: options.remotePort },
  })
}

const full = async (options) => {
  ensureBackendDeps(options)
  ensureFrontendDeps()

  console.log('Starting backend, frontend, and laser pointer remote hub...')
  console.log(`Presentation: http://localhost:${options.frontendPort}`)
  console.log(`Phone remote: http://<laptop-ip>:${options.frontendPort}/remote`)

  const children = [
    spawnService('backend', venvPython(options), [
      '-m',
      'uvicorn',
      'main:app',
      '--reload',
      '--host',
      options.backendHost,
      '--port',
      options.backendPort,
    ], { cwd: backendDir }),
    spawnService('remote', 'npm', ['--prefix', 'presentation', 'run', 'remote'], {
      env: { ...process.env, REMOTE_HOST: options.remoteHost, REMOTE_PORT: options.remotePort },
    }),
    spawnService('frontend', 'npm', ['--prefix', 'presentation', 'run', 'dev', '--', '--host', options.frontendHost, '--port', options.frontendPort], {
      env: { ...process.env, VITE_REMOTE_WS_URL: options.viteRemoteWsUrl },
    }),
  ]

  let stopping = false
  const stop = (code = 0) => {
    if (stopping) return
    stopping = true
    console.log('\nStopping presentation app...')
    for (const child of children) {
      stopChild(child)
    }
    setTimeout(() => process.exit(code), 500)
  }

  process.on('SIGINT', () => stop(0))
  process.on('SIGTERM', () => stop(0))

  for (const child of children) {
    child.on('exit', (code) => {
      if (!stopping) stop(code ?? 1)
    })
  }
}

const main = async () => {
  const { target, options } = parseArgs(process.argv.slice(2))
  const normalizedTarget = target === 'laser' || target === 'remote' ? 'remote' : target
  const targetMap = { help: printHelp, setup, examples, backend, frontend, remote, run: full, full, presentation: full }
  const action = targetMap[normalizedTarget]

  if (!action) {
    throw new Error(`Unknown target: ${target}`)
  }

  await action(options)
}

main().catch((error) => {
  console.error(error.message)
  process.exit(1)
})
