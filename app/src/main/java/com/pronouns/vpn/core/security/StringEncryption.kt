package com.pronouns.vpn.core.security

import android.util.Base64

object StringEncryption {

    fun encrypt(plaintext: String, key: Int): String {
        val bytes = plaintext.toByteArray(Charsets.UTF_8)
        val xored = ByteArray(bytes.size) { i ->
            (bytes[i].toInt() xor (key shr ((i % 4) * 8) and 0xFF)).toByte()
        }
        return Base64.encodeToString(xored, Base64.NO_WRAP)
    }

    fun decrypt(encrypted: String, key: Int): String {
        val bytes = Base64.decode(encrypted, Base64.NO_WRAP)
        val xored = ByteArray(bytes.size) { i ->
            (bytes[i].toInt() xor (key shr ((i % 4) * 8) and 0xFF)).toByte()
        }
        return String(xored, Charsets.UTF_8)
    }
}
