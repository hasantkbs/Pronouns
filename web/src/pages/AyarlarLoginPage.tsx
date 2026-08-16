export default function AyarlarLoginPage({ onSuccess }: { onSuccess: () => void }) {
  return (
    <main>
      <button onClick={onSuccess}>Giriş (yakında)</button>
    </main>
  );
}
