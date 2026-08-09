package com.pronouns.vpn.security

import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class DetectorManager @Inject constructor(
    private val rootDetector: RootDetector,
    private val debuggerDetector: DebuggerDetector,
    private val emulatorDetector: EmulatorDetector,
    private val fridaDetector: FridaDetector
) {
    fun isCompromised(): Boolean {
        val checks = listOf(
            rootDetector.isDeviceRooted(),
            debuggerDetector.isDebuggerAttached(),
            emulatorDetector.isEmulator(),
            fridaDetector.isFridaPresent()
        )

        val compromised = checks.any { it }

        if (compromised) {
            android.util.Log.w("DetectorManager",
                "Device security check failed. Root: ${rootDetector.isDeviceRooted()}, " +
                        "Debugger: ${debuggerDetector.isDebuggerAttached()}, " +
                        "Emulator: ${emulatorDetector.isEmulator()}, " +
                        "Frida: ${fridaDetector.isFridaPresent()}"
            )
        }

        return compromised
    }

    fun getSecurityStatus(): SecurityStatus {
        return SecurityStatus(
            isRooted = rootDetector.isDeviceRooted(),
            isDebuggerAttached = debuggerDetector.isDebuggerAttached(),
            isEmulator = emulatorDetector.isEmulator(),
            isFridaDetected = fridaDetector.isFridaPresent(),
            isCompromised = isCompromised()
        )
    }

    data class SecurityStatus(
        val isRooted: Boolean,
        val isDebuggerAttached: Boolean,
        val isEmulator: Boolean,
        val isFridaDetected: Boolean,
        val isCompromised: Boolean
    )
}
