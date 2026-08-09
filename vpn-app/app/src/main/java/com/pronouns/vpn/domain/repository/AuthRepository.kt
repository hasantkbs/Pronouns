package com.pronouns.vpn.domain.repository

import com.pronouns.vpn.domain.model.AuthCredentials

interface AuthRepository {
    suspend fun getStoredCredentials(): AuthCredentials?
    suspend fun saveCredentials(credentials: AuthCredentials)
    suspend fun clearCredentials()
    suspend fun getApiToken(): String?
    suspend fun saveApiToken(token: String)
    suspend fun clearApiToken()
    suspend fun rotateCredentials(): AuthCredentials
    fun hasCredentials(): Boolean
    fun hasApiToken(): Boolean
}
