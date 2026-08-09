package com.pronouns.vpn.core.detection

import android.content.Context
import android.content.pm.PackageManager
import android.content.pm.Signature
import java.security.MessageDigest
import java.security.cert.Certificate

object IntegrityChecker {

    private const val PLAY_STORE_PACKAGE = "com.android.vending"

    fun verifyIntegrity(context: Context): Boolean {
        if (RootDetector.isRooted()) return false
        if (DebuggerDetector.isDebuggerAttached()) return false
        if (EmulatorDetector.isEmulator()) return false
        if (FridaDetector.isFridaPresent()) return false
        if (!isInstalledFromPlayStore(context)) return false
        if (!isSignatureConsistent(context)) return false
        if (!isApkTampered(context)) return false
        return true
    }

    private fun isInstalledFromPlayStore(context: Context): Boolean {
        return try {
            val installer = context.packageManager
                .getInstallerPackageName(context.packageName)
            installer == PLAY_STORE_PACKAGE
        } catch (_: Exception) {
            false
        }
    }

    private fun isSignatureConsistent(context: Context): Boolean {
        return try {
            val packageInfo = context.packageManager.getPackageInfo(
                context.packageName,
                PackageManager.GET_SIGNING_CERTIFICATES
            )
            val signingInfo = packageInfo.signingInfo ?: return false
            if (signingInfo.hasMultipleSigners()) {
                signingInfo.apkContentsSigners.isNotEmpty()
            } else {
                signingInfo.signingCertificateHistory.isNotEmpty()
            }
        } catch (_: Exception) {
            false
        }
    }

    private fun isApkTampered(context: Context): Boolean {
        return try {
            val packageInfo = context.packageManager.getPackageInfo(
                context.packageName,
                PackageManager.GET_SIGNING_CERTIFICATES
            )
            val signingInfo = packageInfo.signingInfo ?: return false
            val hash = if (signingInfo.hasMultipleSigners()) {
                val sigs = signingInfo.apkContentsSigners
                if (sigs.isEmpty()) return false
                hashSignature(sigs[0])
            } else {
                val certs = signingInfo.signingCertificateHistory
                if (certs.isEmpty()) return false
                hashCertificate(certs[0])
            }
            hash.isNotEmpty()
        } catch (_: Exception) {
            false
        }
    }

    private fun hashSignature(sig: Signature): String {
        val digest = MessageDigest.getInstance("SHA-256")
        return bytesToHex(digest.digest(sig.toByteArray()))
    }

    private fun hashCertificate(cert: Certificate): String {
        val digest = MessageDigest.getInstance("SHA-256")
        return bytesToHex(digest.digest(cert.encoded))
    }

    private fun bytesToHex(bytes: ByteArray): String {
        val hexChars = CharArray(bytes.size * 2)
        for (i in bytes.indices) {
            val v = bytes[i].toInt() and 0xFF
            hexChars[i * 2] = "0123456789ABCDEF"[v ushr 4]
            hexChars[i * 2 + 1] = "0123456789ABCDEF"[v and 0x0F]
        }
        return String(hexChars)
    }
}
