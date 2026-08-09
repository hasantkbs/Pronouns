package com.pronouns.vpn.vpn

import android.content.Context
import com.pronouns.vpn.core.security.StringEncryption
import com.pronouns.vpn.domain.model.VpnProfile
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class VpnProfileLoader @Inject constructor(
    @ApplicationContext private val context: Context
) {

    suspend fun loadEncryptedProfile(): Result<String> = runCatching {
        val encryptedBytes = context.assets.open(ASSET_NAME).use { it.readBytes() }
        val encryptedText = encryptedBytes.toString(Charsets.UTF_8)
        StringEncryption.decrypt(encryptedText, XOR_KEY)
    }

    fun parseProfileConfig(configText: String): VpnProfile {
        val lines = configText.lines()
        var serverHost = ""
        var serverPort = 443
        var protocol = "udp"
        var cipher: String? = null
        val extraOptions = mutableMapOf<String, String>()
        val blockContents = mutableMapOf<String, String>()
        var currentBlock: String? = null
        val currentLines = mutableListOf<String>()

        for (line in lines) {
            val trimmed = line.trim()

            if (trimmed.startsWith("<") && trimmed.endsWith(">") && !trimmed.startsWith("</")) {
                currentBlock = trimmed.removeSurrounding("<", ">")
                currentLines.clear()
                continue
            }

            if (trimmed.startsWith("</") && trimmed.endsWith(">") && currentBlock != null) {
                val closingTag = trimmed.removeSurrounding("</", ">")
                if (closingTag == currentBlock) {
                    blockContents[currentBlock] = currentLines.joinToString("\n")
                    currentBlock = null
                    currentLines.clear()
                }
                continue
            }

            if (currentBlock != null) {
                currentLines.add(line)
                continue
            }

            when {
                trimmed.startsWith("remote ") -> {
                    val parts = trimmed.removePrefix("remote ").split("\\s+".toRegex(), limit = 3)
                    serverHost = parts[0]
                    serverPort = parts.getOrNull(1)?.toIntOrNull() ?: 443
                }
                trimmed.startsWith("proto ") -> {
                    protocol = trimmed.removePrefix("proto ").trim()
                }
                trimmed.startsWith("cipher ") -> {
                    cipher = trimmed.removePrefix("cipher ").trim()
                }
                trimmed.startsWith("#") || trimmed.startsWith(";") || trimmed.isBlank() -> {
                    continue
                }
                trimmed.startsWith("client") || trimmed.startsWith("dev ") ||
                    trimmed.startsWith("resolv-retry") || trimmed.startsWith("nobind") ||
                    trimmed.startsWith("persist-key") || trimmed.startsWith("persist-tun") ||
                    trimmed.startsWith("remote-cert-tls") || trimmed.startsWith("auth ") ||
                    trimmed.startsWith("verb ") || trimmed.startsWith("auth-user-pass") ||
                    trimmed.startsWith("key-direction") -> {
                    continue
                }
                else -> {
                    val spaceIdx = trimmed.indexOf(' ')
                    if (spaceIdx > 0) {
                        extraOptions[trimmed.substring(0, spaceIdx)] =
                            trimmed.substring(spaceIdx + 1).trim()
                    }
                }
            }
        }

        return VpnProfile(
            serverHost = serverHost,
            serverPort = serverPort,
            protocol = protocol,
            caCert = blockContents["ca"],
            clientCert = blockContents["cert"],
            clientKey = blockContents["key"],
            tlsAuth = blockContents["tls-auth"] ?: blockContents["tls-crypt"],
            cipher = cipher,
            extraOptions = extraOptions
        )
    }

    private companion object {
        private const val ASSET_NAME = "encrypted_profile.bin"
        private const val XOR_KEY = 0x3A7B5C9D
    }
}
