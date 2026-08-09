package com.pronouns.vpn.core.utils

object Zeroizer {

    inline fun zeroize(vararg bytes: ByteArray) {
        for (array in bytes) {
            array.fill(0)
        }
    }

    fun zeroizeString(s: String) {
        try {
            val field = String::class.java.getDeclaredField("value")
            field.isAccessible = true
            val value = field.get(s)
            when (value) {
                is CharArray -> value.fill('\u0000')
                is ByteArray -> value.fill(0)
            }
        } catch (_: Exception) {
        }
    }
}
