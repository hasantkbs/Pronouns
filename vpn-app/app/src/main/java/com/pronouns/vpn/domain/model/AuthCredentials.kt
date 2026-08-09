package com.pronouns.vpn.domain.model

import java.time.Instant

data class AuthCredentials(
    val username: String,
    val password: String,
    val expiresAt: Instant,
    val sessionToken: String? = null,
    val isEphemeral: Boolean = true
) {
    fun isExpired(): Boolean = Instant.now().isAfter(expiresAt)

    fun isExpiringWithin(seconds: Long): Boolean =
        Instant.now().plusSeconds(seconds).isAfter(expiresAt)
}
