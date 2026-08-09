package com.pronouns.vpn.domain.model

data class AuthToken(
    val token: String,
    val expiry: Long
)
