package com.pronouns.vpn.domain.repository

import com.pronouns.vpn.domain.model.DeviceInfo

interface DeviceRepository {
    suspend fun getDeviceInfo(): DeviceInfo
    suspend fun isDeviceRegistered(): Boolean
    suspend fun markDeviceRegistered()
    suspend fun getDeviceId(): String
    suspend fun generateAndStoreDeviceId(): String
}
