package com.pronouns.vpn.core.security

import android.util.Base64
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec

class CryptoManager(
    private val androidKeystoreWrapper: AndroidKeystoreWrapper
) {
    private val random = SecureRandom()

    @Synchronized
    fun encrypt(alias: String, plaintext: ByteArray): ByteArray {
        val secretKey = requireNotNull(androidKeystoreWrapper.getKey(alias)) {
            "Key not found: $alias"
        }
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, secretKey, random)
        val iv = cipher.iv
        val ciphertext = cipher.doFinal(plaintext)
        return iv + ciphertext
    }

    @Synchronized
    fun decrypt(alias: String, ciphertext: ByteArray): ByteArray {
        val secretKey = requireNotNull(androidKeystoreWrapper.getKey(alias)) {
            "Key not found: $alias"
        }
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        val iv = ciphertext.copyOfRange(0, GCM_IV_LENGTH)
        val encrypted = ciphertext.copyOfRange(GCM_IV_LENGTH, ciphertext.size)
        val spec = GCMParameterSpec(GCM_TAG_LENGTH, iv)
        cipher.init(Cipher.DECRYPT_MODE, secretKey, spec)
        return cipher.doFinal(encrypted)
    }

    fun encryptString(alias: String, plaintext: String): String {
        val encrypted = encrypt(alias, plaintext.toByteArray(Charsets.UTF_8))
        return Base64.encodeToString(encrypted, Base64.NO_WRAP)
    }

    fun decryptString(alias: String, ciphertext: String): String {
        val encrypted = Base64.decode(ciphertext, Base64.NO_WRAP)
        val decrypted = decrypt(alias, encrypted)
        return String(decrypted, Charsets.UTF_8)
    }

    private companion object {
        private const val GCM_IV_LENGTH = 12
        private const val GCM_TAG_LENGTH = 128
    }
}
