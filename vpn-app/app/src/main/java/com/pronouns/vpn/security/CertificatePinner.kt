package com.pronouns.vpn.security

import android.util.Base64
import java.security.MessageDigest
import java.security.cert.CertificateFactory
import java.security.cert.X509Certificate
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class CertificatePinner @Inject constructor() {

    private val pinnedCertificates = mutableMapOf<String, List<String>>()

    fun addPin(hostname: String, vararg pins: String) {
        pinnedCertificates[hostname] = pins.toList()
    }

    fun verifyCertificateChain(
        hostname: String,
        certificateChain: Array<java.security.cert.Certificate>
    ): Boolean {
        val pins = pinnedCertificates[hostname] ?: return true

        for (cert in certificateChain) {
            val certBytes = cert.encoded
            val digest = MessageDigest.getInstance("SHA-256")
            val hash = digest.digest(certBytes)
            val hashBase64 = Base64.encodeToString(hash, Base64.NO_WRAP)

            for (pin in pins) {
                val pinHash = pin.removePrefix("sha256/")
                if (hashBase64 == pinHash) return true
            }
        }

        android.util.Log.e("CertificatePinner", "Certificate pinning failed for $hostname")
        return false
    }

    fun getCertificateFingerprint(certificate: X509Certificate): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val hash = digest.digest(certificate.encoded)
        val hashHex = hash.joinToString("") { "%02X".format(it) }
        return hashHex.chunked(2).joinToString(":")
    }
}
