package com.pronouns.vpn.domain.model

data class VpnProfile(
    val id: String,
    val name: String,
    val serverAddress: String,
    val serverPort: Int,
    val protocol: Protocol,
    val encryptedConfigData: ByteArray,
    val certificateHash: String
) {
    enum class Protocol { UDP, TCP }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is VpnProfile) return false
        return id == other.id
    }

    override fun hashCode(): Int = id.hashCode()
}
