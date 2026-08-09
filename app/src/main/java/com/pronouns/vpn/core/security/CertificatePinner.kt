package com.pronouns.vpn.core.security

import android.content.Context
import okhttp3.CertificatePinner
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

object CertificatePinnerBuilder {

    private const val PINS_RESOURCE_NAME = "pins"

    fun buildCertificatePinner(context: Context): CertificatePinner {
        val builder = CertificatePinner.Builder()
        val json = readPinsJson(context) ?: return builder.build()
        val pins: JSONArray
        try {
            val root = JSONObject(json)
            if (root.has("expiration") && root.getLong("expiration") < System.currentTimeMillis()) {
                return builder.build()
            }
            pins = root.getJSONArray("pins")
        } catch (_: Exception) {
            try {
                pins = JSONArray(json)
            } catch (_: Exception) {
                return builder.build()
            }
        }
        for (i in 0 until pins.length()) {
            val pinObj = pins.getJSONObject(i)
            val host = pinObj.getString("host")
            val sha256 = pinObj.getString("sha256")
            builder.add(host, "sha256/$sha256")
        }
        return builder.build()
    }

    private fun readPinsJson(context: Context): String? {
        val resId = context.resources.getIdentifier(
            PINS_RESOURCE_NAME,
            "raw",
            context.packageName
        )
        if (resId == 0) return null
        return BufferedReader(InputStreamReader(context.resources.openRawResource(resId)))
            .use { it.readText() }
    }
}
