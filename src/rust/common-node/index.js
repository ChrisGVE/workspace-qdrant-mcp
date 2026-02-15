/* eslint-disable */
/* Native addon loader for wqm-common-node */
/* Based on standard napi-rs loader pattern */

const { existsSync, readFileSync } = require('fs')
const { join } = require('path')
const { execFileSync } = require('child_process')

const { platform, arch } = process

let nativeBinding = null
let localFileExisted = false
let loadError = null

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
  default:
    throw new Error(`Unsupported OS: ${platform}, architecture: ${arch}`)
}

if (!nativeBinding) {
  if (loadError) {
    throw loadError
  }
  throw new Error(`Failed to load native binding`)
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
module.exports.collectionMemory = nativeBinding.collectionMemory
module.exports.collectionScratchpad = nativeBinding.collectionScratchpad
module.exports.defaultQdrantUrl = nativeBinding.defaultQdrantUrl
module.exports.defaultGrpcPort = nativeBinding.defaultGrpcPort
module.exports.defaultBranch = nativeBinding.defaultBranch
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
