# Keep Hilt
-keep class dagger.hilt.** { *; }
-keep class javax.inject.** { *; }
-keep class * extends dagger.hilt.android.internal.managers.ViewComponentManager$FragmentContextWrapper { *; }

# Keep Moshi
-keep class com.pronouns.vpn.data.remote.dto.** { *; }
-keep class com.pronouns.vpn.domain.model.** { *; }
-keep class com.squareup.moshi.** { *; }
-keep @com.squareup.moshi.JsonClass class * { *; }

# Keep Retrofit
-keep class retrofit2.** { *; }
-keepclassmembers,allowshrinking,allowobfuscation interface * {
    @retrofit2.http.* <methods>;
}
-dontwarn retrofit2.**
-keep,includedescriptorclasses class com.pronouns.vpn.data.remote.api.** { *; }

# Keep OkHttp
-keep class okhttp3.** { *; }
-keep interface okhttp3.** { *; }
-dontwarn okhttp3.**
-dontwarn okio.**

# Keep OpenVPN
-keep class net.openvpn.** { *; }
-dontwarn net.openvpn.**

# Keep WorkManager
-keep class * extends androidx.work.Worker {
    public <init>(...);
}

# Keep Coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembers class kotlinx.coroutines.** {
    volatile <fields>;
}

# Keep AndroidX Security
-keep class androidx.security.crypto.** { *; }

# Keep DataStore
-keep class androidx.datastore.** { *; }

# Remove logging in release
-assumenosideeffects class android.util.Log {
    public static boolean isLoggable(java.lang.String, int);
    public static int v(...);
    public static int d(...);
    public static int i(...);
}

# String encryption obfuscation
-keepclassmembers class com.pronouns.vpn.security.MemoryZeroizer {
    private static void zeroize(...);
}

# Keep application
-keep class com.pronouns.vpn.VpnApplication { *; }

# Keep BuildConfig
-keep class com.pronouns.vpn.BuildConfig { *; }

# Moshi adapters
-keep class com.squareup.moshi.** { *; }
-keepclassmembers class * {
    @com.squareup.moshi.FromJson <methods>;
    @com.squareup.moshi.ToJson <methods>;
}

# Enum classes
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Parcelable
-keepclassmembers class * implements android.os.Parcelable {
    public static final android.os.Parcelable$Creator CREATOR;
}

# R8 full mode exceptions
-keepattributes Signature, InnerClasses, EnclosingMethod, *Annotation*
-keepattributes RuntimeVisibleAnnotations, RuntimeVisibleParameterAnnotations
-keepattributes RuntimeInvisibleAnnotations, RuntimeInvisibleParameterAnnotations
-keepattributes SourceFile, LineNumberTable
-keepattributes Exceptions

# Kotlin
-keepclassmembers class kotlin.Metadata { *; }
-dontwarn kotlin.**

# Moshi Sealed classes
-keep class kotlin.reflect.** { *; }
