/* eslint-disable */
/* Native addon loader for wqm-common-node */
/* Based on standard napi-rs loader pattern */

const { existsSync, readFileSync } = require('fs')
const { createHash } = require('crypto')
const { homedir } = require('os')
const { join, resolve } = require('path')
const { execFileSync } = require('child_process')

const { platform, arch } = process

let nativeBinding = null
let localFileExisted = false
let loadError = null

function loadDllBinding(bindingPath) {
  const addonModule = { exports: {} }
  process.dlopen(addonModule, bindingPath)
  return addonModule.exports
}

function isMusl() {
  if (!process.report || typeof process.report.getReport !== 'function') {
    try {
      const lddPath = execFileSync('which', ['ldd']).toString().trim()
      return readFileSync(lddPath, 'utf8').includes('musl')
    } catch {
      return true
    }
  } else {
    const { glibcVersionRuntime } = process.report.getReport().header
    return !glibcVersionRuntime
  }
}

function sha256Hex(value) {
  return createHash('sha256').update(value).digest('hex')
}

function normalizeGitUrl(url) {
  let normalized = String(url).toLowerCase()

  for (const protocol of ['https://', 'http://', 'ssh://', 'git://']) {
    if (normalized.startsWith(protocol)) {
      normalized = normalized.slice(protocol.length)
      break
    }
  }

  if (normalized.startsWith('git@')) {
    normalized = normalized.slice(4)
    const idx = normalized.indexOf(':')
    if (idx !== -1) {
      normalized = `${normalized.slice(0, idx)}/${normalized.slice(idx + 1)}`
    }
  }

  if (normalized.endsWith('.git')) {
    normalized = normalized.slice(0, -4)
  }

  return normalized.replace(/\/+$/g, '')
}

function normalizeCanonicalPath(input) {
  if (!input) {
    throw new Error('path cannot be empty')
  }

  if (String(input).includes('\0')) {
    throw new Error('path contains embedded NUL byte')
  }

  let expanded = String(input)
  if (expanded.startsWith('~')) {
    expanded = join(homedir(), expanded.slice(1))
  }

  // Keep the fallback aligned with the Rust canonical path rules.
  const asPosix = expanded.replace(/\\/g, '/')
  if (!asPosix.startsWith('/')) {
    throw new Error('relative input')
  }

  const parts = []
  for (const component of asPosix.split('/')) {
    if (!component || component === '.') {
      continue
    }
    if (component === '..') {
      throw new Error('path contains parent directory component')
    }
    parts.push(component)
  }

  return `/${parts.join('/')}`
}

function pathToSyntacticCanonical(pathValue) {
  try {
    return normalizeCanonicalPath(pathValue)
  } catch {
    try {
      return normalizeCanonicalPath(resolve(process.cwd(), String(pathValue)))
    } catch {
      return String(pathValue)
    }
  }
}

function calculateProjectIdJs(projectRoot, gitRemote) {
  if (gitRemote) {
    const normalized = normalizeGitUrl(gitRemote)
    return sha256Hex(normalized).slice(0, 12)
  }

  return `local_${sha256Hex(pathToSyntacticCanonical(projectRoot)).slice(0, 12)}`
}

function calculateProjectIdWithDisambiguationJs(projectRoot, gitRemote, disambiguationPath) {
  if (gitRemote) {
    const normalized = normalizeGitUrl(gitRemote)
    const input = disambiguationPath ? `${normalized}|${disambiguationPath}` : normalized
    return sha256Hex(input).slice(0, 12)
  }

  return `local_${sha256Hex(pathToSyntacticCanonical(projectRoot)).slice(0, 12)}`
}

function detectGitRemoteJs(projectRoot) {
  for (const remoteName of ['origin', 'upstream']) {
    try {
      const output = execFileSync(
        'git',
        ['-C', String(projectRoot), 'remote', 'get-url', remoteName],
        { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }
      ).trim()

      if (output) {
        return output
      }
    } catch {
      // keep trying other remotes
    }
  }
  return null
}

function generateIdempotencyKeyJs(itemType, op, tenantId, collection, payloadJson) {
  if (!tenantId || !collection) {
    return null
  }

  if (!isValidItemTypeJs(itemType) || !isValidQueueOperationJs(op) || !isValidOperationForTypeJs(itemType, op)) {
    return null
  }

  return sha256Hex(`${itemType}|${op}|${tenantId}|${collection}|${payloadJson}`).slice(0, 32)
}

function computeContentHashJs(content) {
  return sha256Hex(String(content))
}

function tokenizeJs(text) {
  const stopwords = new Set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
    'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the', 'to',
    'was', 'were', 'will', 'with', 'this', 'these', 'those', 'you', 'your',
  ])

  return String(text)
    .toLowerCase()
    .match(/[a-z0-9]+/g)?.filter((token) => token.length > 1 && !stopwords.has(token)) ?? []
}

function isValidItemTypeJs(value) {
  return [
    'text',
    'file',
    'url',
    'website',
    'doc',
    'folder',
    'tenant',
    'collection',
  ].includes(value)
}

function isValidQueueOperationJs(value) {
  return ['add', 'update', 'delete', 'scan', 'rename', 'uplift', 'reset', 'reembed'].includes(value)
}

function isValidQueueStatusJs(value) {
  return ['pending', 'in_progress', 'done', 'failed'].includes(value)
}

function isValidOperationForTypeJs(itemType, op) {
  const valid = {
    text: ['add', 'update', 'delete', 'uplift'],
    file: ['add', 'update', 'delete', 'rename', 'uplift'],
    url: ['add', 'update', 'delete', 'uplift'],
    website: ['add', 'update', 'delete', 'scan', 'uplift'],
    doc: ['delete', 'uplift'],
    folder: ['delete', 'scan', 'rename'],
    tenant: ['add', 'update', 'delete', 'scan', 'rename', 'uplift'],
    collection: ['uplift', 'reset', 'reembed'],
  }

  return valid[itemType]?.includes(op) ?? false
}

function createJsFallbackBinding() {
  return {
    calculateProjectId: calculateProjectIdJs,
    calculateProjectIdWithDisambiguation: calculateProjectIdWithDisambiguationJs,
    normalizeGitUrl,
    detectGitRemote: detectGitRemoteJs,
    calculateTenantId(projectRoot) {
      return calculateProjectIdJs(projectRoot, detectGitRemoteJs(projectRoot))
    },
    generateIdempotencyKey: generateIdempotencyKeyJs,
    computeContentHash: computeContentHashJs,
    tokenize: tokenizeJs,
    collectionProjects: () => 'projects',
    collectionLibraries: () => 'libraries',
    collectionRules: () => 'rules',
    collectionScratchpad: () => 'scratchpad',
    defaultQdrantUrl: () => 'http://localhost:6333',
    defaultGrpcPort: () => 50051,
    defaultBranch: () => 'main',
    priorityHigh: () => 1,
    priorityNormal: () => 3,
    priorityLow: () => 5,
    fieldTenantId: () => 'tenant_id',
    fieldProjectId: () => 'project_id',
    fieldLibraryName: () => 'library_name',
    fieldBasePoint: () => 'base_point',
    fieldBranch: () => 'branch',
    fieldFileType: () => 'file_type',
    fieldFilePath: () => 'file_path',
    fieldConceptTags: () => 'concept_tags',
    fieldDeleted: () => 'deleted',
    fieldContent: () => 'content',
    fieldTitle: () => 'title',
    fieldSourceType: () => 'source_type',
    fieldDocumentId: () => 'document_id',
    fieldItemType: () => 'item_type',
    fieldParentUnitId: () => 'parent_unit_id',
    itemTypeText: () => 'text',
    itemTypeFile: () => 'file',
    itemTypeUrl: () => 'url',
    itemTypeWebsite: () => 'website',
    itemTypeDoc: () => 'doc',
    itemTypeFolder: () => 'folder',
    itemTypeTenant: () => 'tenant',
    itemTypeCollection: () => 'collection',
    allItemTypes: () => ['text', 'file', 'url', 'website', 'doc', 'folder', 'tenant', 'collection'],
    operationAdd: () => 'add',
    operationUpdate: () => 'update',
    operationDelete: () => 'delete',
    operationScan: () => 'scan',
    operationRename: () => 'rename',
    operationUplift: () => 'uplift',
    operationReset: () => 'reset',
    allOperations: () => ['add', 'update', 'delete', 'scan', 'rename', 'uplift', 'reset', 'reembed'],
    isValidItemType: isValidItemTypeJs,
    isValidQueueOperation: isValidQueueOperationJs,
    isValidQueueStatus: isValidQueueStatusJs,
    isValidOperationForType: isValidOperationForTypeJs,
  }
}

switch (platform) {
  case 'darwin':
    switch (arch) {
      case 'x64':
        localFileExisted = existsSync(join(__dirname, 'wqm-common-node.darwin-x64.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./wqm-common-node.darwin-x64.node')
          } else {
            nativeBinding = require('wqm-common-node-darwin-x64')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm64':
        localFileExisted = existsSync(join(__dirname, 'wqm-common-node.darwin-arm64.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./wqm-common-node.darwin-arm64.node')
          } else {
            nativeBinding = require('wqm-common-node-darwin-arm64')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on macOS: ${arch}`)
    }
    break
  case 'linux':
    switch (arch) {
      case 'x64':
        if (isMusl()) {
          localFileExisted = existsSync(join(__dirname, 'wqm-common-node.linux-x64-musl.node'))
          try {
            if (localFileExisted) {
              nativeBinding = require('./wqm-common-node.linux-x64-musl.node')
            } else {
              nativeBinding = require('wqm-common-node-linux-x64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(join(__dirname, 'wqm-common-node.linux-x64-gnu.node'))
          try {
            if (localFileExisted) {
              nativeBinding = require('./wqm-common-node.linux-x64-gnu.node')
            } else {
              nativeBinding = require('wqm-common-node-linux-x64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 'arm64':
        if (isMusl()) {
          localFileExisted = existsSync(join(__dirname, 'wqm-common-node.linux-arm64-musl.node'))
          try {
            if (localFileExisted) {
              nativeBinding = require('./wqm-common-node.linux-arm64-musl.node')
            } else {
              nativeBinding = require('wqm-common-node-linux-arm64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(join(__dirname, 'wqm-common-node.linux-arm64-gnu.node'))
          try {
            if (localFileExisted) {
              nativeBinding = require('./wqm-common-node.linux-arm64-gnu.node')
            } else {
              nativeBinding = require('wqm-common-node-linux-arm64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      default:
        throw new Error(`Unsupported architecture on Linux: ${arch}`)
    }
    break
  case 'win32':
    switch (arch) {
      case 'x64':
        localFileExisted = existsSync(join(__dirname, 'wqm-common-node.win32-x64-msvc.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./wqm-common-node.win32-x64-msvc.node')
          } else {
            const releaseBinding = join(
              __dirname,
              '..',
              'target',
              'x86_64-pc-windows-msvc',
              'release',
              'wqm_common_node.dll'
            )
            const debugBinding = join(
              __dirname,
              '..',
              'target',
              'x86_64-pc-windows-msvc',
              'debug',
              'wqm_common_node.dll'
            )
            if (existsSync(releaseBinding)) {
              nativeBinding = loadDllBinding(releaseBinding)
            } else if (existsSync(debugBinding)) {
              nativeBinding = loadDllBinding(debugBinding)
            } else {
              nativeBinding = require('wqm-common-node-win32-x64-msvc')
            }
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm64':
        localFileExisted = existsSync(join(__dirname, 'wqm-common-node.win32-arm64-msvc.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./wqm-common-node.win32-arm64-msvc.node')
          } else {
            const releaseBinding = join(
              __dirname,
              '..',
              'target',
              'aarch64-pc-windows-msvc',
              'release',
              'wqm_common_node.dll'
            )
            const debugBinding = join(
              __dirname,
              '..',
              'target',
              'aarch64-pc-windows-msvc',
              'debug',
              'wqm_common_node.dll'
            )
            if (existsSync(releaseBinding)) {
              nativeBinding = loadDllBinding(releaseBinding)
            } else if (existsSync(debugBinding)) {
              nativeBinding = loadDllBinding(debugBinding)
            } else {
              nativeBinding = require('wqm-common-node-win32-arm64-msvc')
            }
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Windows: ${arch}`)
    }
    break
  default:
    throw new Error(`Unsupported OS: ${platform}, architecture: ${arch}`)
}

if (!nativeBinding) {
  if (loadError) {
    nativeBinding = createJsFallbackBinding()
  } else {
    throw new Error(`Failed to load native binding`)
  }
}

module.exports.calculateProjectId = nativeBinding.calculateProjectId
module.exports.calculateProjectIdWithDisambiguation = nativeBinding.calculateProjectIdWithDisambiguation
module.exports.normalizeGitUrl = nativeBinding.normalizeGitUrl
module.exports.detectGitRemote = nativeBinding.detectGitRemote
module.exports.calculateTenantId = nativeBinding.calculateTenantId
module.exports.generateIdempotencyKey = nativeBinding.generateIdempotencyKey
module.exports.computeContentHash = nativeBinding.computeContentHash
module.exports.tokenize = nativeBinding.tokenize
module.exports.collectionProjects = nativeBinding.collectionProjects
module.exports.collectionLibraries = nativeBinding.collectionLibraries
module.exports.collectionRules = nativeBinding.collectionRules
module.exports.collectionScratchpad = nativeBinding.collectionScratchpad
module.exports.defaultQdrantUrl = nativeBinding.defaultQdrantUrl
module.exports.defaultGrpcPort = nativeBinding.defaultGrpcPort
module.exports.defaultBranch = nativeBinding.defaultBranch
module.exports.fieldTenantId = nativeBinding.fieldTenantId
module.exports.fieldProjectId = nativeBinding.fieldProjectId
module.exports.fieldLibraryName = nativeBinding.fieldLibraryName
module.exports.fieldBasePoint = nativeBinding.fieldBasePoint
module.exports.fieldBranch = nativeBinding.fieldBranch
module.exports.fieldFileType = nativeBinding.fieldFileType
module.exports.fieldFilePath = nativeBinding.fieldFilePath
module.exports.fieldConceptTags = nativeBinding.fieldConceptTags
module.exports.fieldDeleted = nativeBinding.fieldDeleted
module.exports.fieldContent = nativeBinding.fieldContent
module.exports.fieldTitle = nativeBinding.fieldTitle
module.exports.fieldSourceType = nativeBinding.fieldSourceType
module.exports.fieldDocumentId = nativeBinding.fieldDocumentId
module.exports.fieldItemType = nativeBinding.fieldItemType
module.exports.fieldParentUnitId = nativeBinding.fieldParentUnitId
module.exports.isValidItemType = nativeBinding.isValidItemType
module.exports.isValidQueueOperation = nativeBinding.isValidQueueOperation
module.exports.isValidQueueStatus = nativeBinding.isValidQueueStatus
module.exports.isValidOperationForType = nativeBinding.isValidOperationForType
module.exports.priorityHigh = nativeBinding.priorityHigh
module.exports.priorityNormal = nativeBinding.priorityNormal
module.exports.priorityLow = nativeBinding.priorityLow
module.exports.itemTypeText = nativeBinding.itemTypeText
module.exports.itemTypeFile = nativeBinding.itemTypeFile
module.exports.itemTypeUrl = nativeBinding.itemTypeUrl
module.exports.itemTypeWebsite = nativeBinding.itemTypeWebsite
module.exports.itemTypeDoc = nativeBinding.itemTypeDoc
module.exports.itemTypeFolder = nativeBinding.itemTypeFolder
module.exports.itemTypeTenant = nativeBinding.itemTypeTenant
module.exports.itemTypeCollection = nativeBinding.itemTypeCollection
module.exports.allItemTypes = nativeBinding.allItemTypes
module.exports.operationAdd = nativeBinding.operationAdd
module.exports.operationUpdate = nativeBinding.operationUpdate
module.exports.operationDelete = nativeBinding.operationDelete
module.exports.operationScan = nativeBinding.operationScan
module.exports.operationRename = nativeBinding.operationRename
module.exports.operationUplift = nativeBinding.operationUplift
module.exports.operationReset = nativeBinding.operationReset
module.exports.allOperations = nativeBinding.allOperations
