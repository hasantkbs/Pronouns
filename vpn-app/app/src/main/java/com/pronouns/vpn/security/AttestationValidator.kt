package com.pronouns.vpn.security

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.Signature
import java.security.cert.X509Certificate
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AttestationValidator @Inject constructor() {

    private val keyStore by lazy {
        KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
    }

    fun generateAttestationKey(
        context: Context,
        alias: String,
        challenge: ByteArray
    ): Boolean {
        return try {
            val keyPairGenerator = KeyPairGenerator.getInstance(
                KeyProperties.KEY_ALGORITHM_EC,
                "AndroidKeyStore"
            )

            val spec = KeyGenParameterSpec.Builder(
                alias,
                KeyProperties.PURPOSE_SIGN
            )
                .setAlgorithmParameterSpec(
                    java.security.spec.ECGenParameterSpec("secp256r1")
                )
                .setDigests(KeyProperties.DIGEST_SHA256)
                .setAttestationChallenge(challenge)
                .build()

            keyPairGenerator.initialize(spec)
            keyPairGenerator.generateKeyPair()
            true
        } catch (e: Exception) {
            android.util.Log.e("Attestation", "Failed to generate attestation key", e)
            false
        }
    }

    fun getAttestationCertificateChain(alias: String): Array<X509Certificate>? {
        return try {
            val entry = keyStore.getEntry(alias, null) as? KeyStore.PrivateKeyEntry
                ?: return null
            entry.certificateChain.map { it as X509Certificate }.toTypedArray()
        } catch (e: Exception) {
            null
        }
    }

    fun verifyAttestationSignature(
        alias: String,
        data: ByteArray,
        signature: ByteArray
    ): Boolean {
        return try {
            val entry = keyStore.getEntry(alias, null) as? KeyStore.PrivateKeyEntry
                ?: return false
            val cert = entry.certificate as X509Certificate
            val sig = Signature.getInstance("SHA256withECDSA")
            sig.initVerify(cert.publicKey)
            sig.update(data)
            sig.verify(signature)
        } catch (e: Exception) {
            false
        }
    }

    fun createAttestationPayload(
        alias: String,
        nonce: ByteArray
    ): AttestationPayload? {
        return try {
            val entry = keyStore.getEntry(alias, null) as? KeyStore.PrivateKeyEntry
                ?: return null

            val signature = Signature.getInstance("SHA256withECDSA")
            signature.initSign(entry.privateKey)
            signature.update(nonce)
            val sigBytes = signature.sign()

            AttestationPayload(
                certificateChain = entry.certificateChain.map {
                    Base64.encodeToString(it.encoded, Base64.NO_WRAP)
                },
                signature = Base64.encodeToString(sigBytes, Base64.NO_WRAP),
                nonce = Base64.encodeToString(nonce, Base64.NO_WRAP)
            )
        } catch (e: Exception) {
            null
        }
    }

    fun deleteAttestationKey(alias: String) {
        try {
            keyStore.deleteEntry(alias)
        } catch (e: Exception) {
            android.util.Log.e("Attestation", "Failed to delete key", e)
        }
    }

    data class AttestationPayload(
        val certificateChain: List<String>,
        val signature: String,
        val nonce: String
    )
}
