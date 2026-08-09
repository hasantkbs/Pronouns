package com.pronouns.vpn.security

import java.nio.ByteBuffer
import java.nio.CharBuffer
import java.security.SecureRandom
import java.util.Arrays
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class MemoryZeroizer @Inject constructor() {

    private val secureRandom = SecureRandom()

    fun zeroize(byteArray: ByteArray) {
        if (byteArray.isEmpty()) return
        secureRandom.nextBytes(byteArray)
        Arrays.fill(byteArray, 0.toByte())
    }

    fun zeroize(charArray: CharArray) {
        if (charArray.isEmpty()) return
        Arrays.fill(charArray, '\u0000')
    }

    fun zeroize(byteBuffer: ByteBuffer) {
        if (!byteBuffer.hasArray()) return
        byteBuffer.clear()
        val array = ByteArray(byteBuffer.capacity())
        secureRandom.nextBytes(array)
        byteBuffer.put(array)
        byteBuffer.clear()
    }

    fun zeroize(charBuffer: CharBuffer) {
        if (!charBuffer.hasArray()) return
        charBuffer.clear()
        Arrays.fill(charBuffer.array(), '\u0000')
        charBuffer.clear()
    }

    fun zeroizeOnReturn(value: String): String {
        val chars = value.toCharArray()
        val result = String(chars)
        zeroize(chars)
        return result
    }

    fun secureString(value: String): SecureString {
        return SecureString(value.toCharArray(), this)
    }

    fun wipeByteBuffer(buffer: ByteBuffer) {
        if (buffer.isDirect) {
            buffer.clear()
            val randBytes = ByteArray(buffer.capacity())
            secureRandom.nextBytes(randBytes)
            buffer.put(randBytes)
            buffer.clear()
        } else {
            zeroize(buffer)
        }
    }

    class SecureString(
        private val chars: CharArray,
        private val zeroizer: MemoryZeroizer
    ) : AutoCloseable {

        private val stringValue: String = String(chars)

        fun value(): String = stringValue

        override fun close() {
            zeroizer.zeroize(chars)
        }

        protected fun finalize() {
            zeroizer.zeroize(chars)
        }
    }
}
