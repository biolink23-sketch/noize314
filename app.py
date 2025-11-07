import streamlit as st
import numpy as np
import base64
from io import BytesIO
import wave

st.set_page_config(page_title="Pink Noise Generator", page_icon="🎵", layout="centered")

def generate_pink_noise(duration, sample_rate=22050):
    """Generate pink noise using the Voss-McCartney algorithm"""
    n_samples = int(duration * sample_rate)
    n_rows = 16  # number of random sources to add
    
    array = np.zeros((n_rows, n_samples))
    for i in range(n_rows):
        array[i] = np.random.randn(n_samples)
    
    # Scale each row by frequency
    pink = np.zeros(n_samples)
    for i in range(n_rows):
        pink += array[i] * (0.5 ** (i/2.0))
    
    # Normalize
    pink = pink / np.max(np.abs(pink))
    return pink

def create_wav_file(audio_data, sample_rate):
    """Create WAV file from audio data"""
    buffer = BytesIO()
    
    # Scale to 16-bit integer range
    scaled = np.int16(audio_data * 32767)
    
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(scaled.tobytes())
    
    buffer.seek(0)
    return buffer

st.title("🎵 Pink Noise Generator")
st.markdown("Generate pink noise (1/f noise) for meditation, sleep, and focus.")

# Controls
col1, col2 = st.columns(2)

with col1:
    duration = st.slider("Duration (seconds)", 5, 30, 10)
    volume = st.slider("Volume", 0.1, 1.0, 0.5)

with col2:
    quality = st.radio("Quality", ["Standard (22 kHz)", "High (44 kHz)"])
    sample_rate = 22050 if "Standard" in quality else 44100

if st.button("🎲 Generate Pink Noise", type="primary"):
    with st.spinner("Generating..."):
        # Generate noise
        noise = generate_pink_noise(duration, sample_rate)
        noise = noise * volume
        
        # Create WAV file
        wav_buffer = create_wav_file(noise, sample_rate)
        
        # Create audio player
        audio_bytes = wav_buffer.getvalue()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        st.success("✅ Pink noise generated!")
        
        # Audio player
        st.markdown("### 🎧 Listen")
        audio_html = f'<audio controls autoplay><source src="data:audio/wav;base64,{audio_b64}" type="audio/wav"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
        
        # Simple visualization using Streamlit's native chart
        st.markdown("### 📊 Waveform (first 2 seconds)")
        samples_to_show = min(sample_rate * 2, len(noise))
        st.line_chart(noise[:samples_to_show])
        
        # Download button
        st.markdown("### 💾 Download")
        st.download_button(
            label="📥 Download WAV File",
            data=wav_buffer,
            file_name=f"pink_noise_{duration}s.wav",
            mime="audio/wav"
        )

with st.expander("ℹ️ About Pink Noise"):
    st.markdown("""
    **Pink noise** has equal energy per octave, making it sound more natural than white noise.
    
    **Benefits:**
    - 🧘 Improves meditation and relaxation
    - 😴 Helps with sleep by masking distracting sounds
    - 🎯 Enhances focus and concentration
    - 🧠 May improve memory consolidation during sleep
    
    **Natural examples:** rain, waterfalls, ocean waves, rustling leaves
    
    **Usage tips:**
    - Start with low volume and adjust to comfort
    - Use for 10-30 minutes for meditation
    - Can be played all night for sleep
    """)

st.markdown("---")
st.caption("Made with ❤️ for better meditation and sleep")
