/**
 * Dubber Studio — Frontend Application Logic
 * Server-Sent Events (SSE) streaming, interactive UI, and HTML5 video player.
 */

// Global Studio Functions for instant click responsiveness
window.toggleStudioConfig = function() {
  const configPanel = document.getElementById('configPanel');
  const configToggleArrow = document.getElementById('configToggleArrow');
  if (!configPanel) return;
  const isExpanded = configPanel.classList.toggle('expanded');
  configPanel.classList.toggle('open', isExpanded);
  if (configToggleArrow) {
    configToggleArrow.classList.toggle('rotated', isExpanded);
    configToggleArrow.classList.toggle('open', isExpanded);
  }
};

window.switchStudioTab = function(tabId, clickedBtn) {
  const allTabBtns = document.querySelectorAll('.studio-tab-btn');
  const allTabContents = document.querySelectorAll('.tab-content');
  allTabBtns.forEach((b) => b.classList.remove('active'));
  allTabContents.forEach((c) => c.classList.remove('active'));
  if (clickedBtn) {
    clickedBtn.classList.add('active');
  } else {
    const matchingBtn = document.querySelector(`.studio-tab-btn[data-tab="${tabId}"]`);
    if (matchingBtn) matchingBtn.classList.add('active');
  }
  const target = document.getElementById(tabId);
  if (target) target.classList.add('active');
};

document.addEventListener('DOMContentLoaded', () => {
  // --- DOM Elements ---
  const dubForm = document.getElementById('dubForm');
  const videoUrlInput = document.getElementById('videoUrlInput');
  const pasteBtn = document.getElementById('pasteBtn');
  const startDubBtn = document.getElementById('startDubBtn');
  const btnText = startDubBtn.querySelector('.btn-text');
  const btnSpinner = startDubBtn.querySelector('.btn-spinner');

  // Language Bridge Elements
  const sourceLangSelect = document.getElementById('sourceLangSelect');
  const targetLangSelect = document.getElementById('targetLangSelect');
  const swapLangBtn = document.getElementById('swapLangBtn');
  const langDirectionBadge = document.getElementById('langDirectionBadge');

  // Studio Tabs
  const tabButtons = document.querySelectorAll('.studio-tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  // Config Elements
  const toggleConfigBtn = document.getElementById('toggleConfigBtn');
  const configPanel = document.getElementById('configPanel');
  const configToggleArrow = document.getElementById('configToggleArrow');
  const whisperModelSelect = document.getElementById('whisperModelSelect');
  const autoGenderToggle = document.getElementById('autoGenderToggle');
  const speakerModeSelect = document.getElementById('speakerModeSelect');
  const maleVoiceSelect = document.getElementById('maleVoiceSelect');
  const femaleVoiceSelect = document.getElementById('femaleVoiceSelect');
  const singleVoiceSelect = document.getElementById('singleVoiceSelect');
  const maleVoiceGroup = document.getElementById('maleVoiceGroup');
  const femaleVoiceGroup = document.getElementById('femaleVoiceGroup');
  const singleVoiceGroup = document.getElementById('singleVoiceGroup');
  const mixOriginalSlider = document.getElementById('mixOriginalSlider');
  const mixValBadge = document.getElementById('mixValBadge');
  const embedSubtitlesToggle = document.getElementById('embedSubtitlesToggle');
  const subtitleModeSelect = document.getElementById('subtitleModeSelect');
  const lipSyncModeSelect = document.getElementById('lipSyncModeSelect');

  // Pipeline Stepper Elements
  const pipelineSection = document.getElementById('pipelineSection');
  const activeJobId = document.getElementById('activeJobId');
  const overallTimer = document.getElementById('overallTimer');
  const pipelineErrorBanner = document.getElementById('pipelineErrorBanner');
  const errorTitle = document.getElementById('errorTitle');
  const errorDetails = document.getElementById('errorDetails');

  // Showcase & Player Elements
  const showcaseSection = document.getElementById('showcaseSection');
  const mainVideoPlayer = document.getElementById('mainVideoPlayer');
  const subtitlesTrack = document.getElementById('subtitlesTrack');
  const playerVideoTitle = document.getElementById('playerVideoTitle');
  const playerVideoMeta = document.getElementById('playerVideoMeta');
  const downloadVideoBtn = document.getElementById('downloadVideoBtn');
  const downloadSrtBtn = document.getElementById('downloadSrtBtn');
  const downloadVttBtn = document.getElementById('downloadVttBtn');
  const downloadBilingualBtn = document.getElementById('downloadBilingualBtn');
  const openProofreadFromPlayerBtn = document.getElementById('openProofreadFromPlayerBtn');
  const deleteCurrentVideoBtn = document.getElementById('deleteCurrentVideoBtn');
  const openProofreadManualBtn = document.getElementById('openProofreadManualBtn');
  const syncedTranscriptList = document.getElementById('syncedTranscriptList');

  // Metrics
  const metricSourceLang = document.getElementById('metricSourceLang');
  const metricLang = document.getElementById('metricLang');
  const metricSpeakers = document.getElementById('metricSpeakers');
  const metricSegments = document.getElementById('metricSegments');
  const metricTotalTime = document.getElementById('metricTotalTime');
  const metricEfficiency = document.getElementById('metricEfficiency');

  // Segments Table Elements
  const segmentsSection = document.getElementById('segmentsSection');
  const segmentsTableBody = document.getElementById('segmentsTableBody');
  const segmentCountBadge = document.getElementById('segmentCountBadge');

  // Proofreading Modal Elements
  const proofreadModal = document.getElementById('proofreadModal');
  const closeProofreadBtn = document.getElementById('closeProofreadBtn');
  const cancelProofreadBtn = document.getElementById('cancelProofreadBtn');
  const saveProofreadBtn = document.getElementById('saveProofreadBtn');
  const redubModalBtn = document.getElementById('redubModalBtn');
  const redubBtnText = document.getElementById('redubBtnText');
  const redubBtnSpinner = document.getElementById('redubBtnSpinner');
  const autoOptimizeAllBtn = document.getElementById('autoOptimizeAllBtn');
  const proofreadCountBadge = document.getElementById('proofreadCountBadge');
  const proofreadList = document.getElementById('proofreadList');

  // Gallery
  const galleryGrid = document.getElementById('galleryGrid');
  const refreshGalleryBtn = document.getElementById('refreshGalleryBtn');

  // State
  let currentEventSource = null;
  let timerInterval = null;
  let timerSeconds = 0;
  let allLanguages = [];
  let currentSegments = [];
  let currentVideoUrl = null;
  let currentSrtUrl = null;
  let currentVttUrl = null;
  let currentBilingualSrtUrl = null;

  // --- Initial Data Loading ---
  loadLanguages();
  loadAvailableVoices();
  loadVideoGallery();

  if (refreshGalleryBtn) {
    refreshGalleryBtn.addEventListener('click', () => {
      loadVideoGallery();
    });
  }

  if (deleteCurrentVideoBtn) {
    deleteCurrentVideoBtn.addEventListener('click', () => {
      if (!currentVideoUrl) return;
      const filename = currentVideoUrl.split('?')[0].split('/').pop();
      deleteDubbedVideo(filename);
    });
  }

  // --- Event Listeners ---

  // Language Bridge & Direction Update
  if (sourceLangSelect && targetLangSelect) {
    sourceLangSelect.addEventListener('change', updateLangDirectionBadge);
    targetLangSelect.addEventListener('change', () => {
      updateLangDirectionBadge();
      updateVoiceOptionsForTarget(targetLangSelect.value);
    });
  }

  if (swapLangBtn) {
    swapLangBtn.addEventListener('click', () => {
      const src = sourceLangSelect.value;
      const tgt = targetLangSelect.value;
      if (src !== 'auto') {
        sourceLangSelect.value = tgt;
        targetLangSelect.value = src;
      } else {
        sourceLangSelect.value = tgt;
        targetLangSelect.value = 'en';
      }
      updateLangDirectionBadge();
      updateVoiceOptionsForTarget(targetLangSelect.value);
    });
  }

  function updateLangDirectionBadge() {
    if (!langDirectionBadge) return;
    const srcText = sourceLangSelect.options[sourceLangSelect.selectedIndex]?.text.split('(')[0].trim() || 'Auto';
    const tgtText = targetLangSelect.options[targetLangSelect.selectedIndex]?.text.split('(')[0].trim() || 'English';
    langDirectionBadge.textContent = `${srcText} ➔ ${tgtText}`;
  }

  // Studio Configuration Tabs
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const targetTabId = btn.dataset.tab;
      const targetContent = document.getElementById(targetTabId);
      if (targetContent) targetContent.classList.add('active');
    });
  });

  // Paste from clipboard
  pasteBtn.addEventListener('click', async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) {
        videoUrlInput.value = text.trim();
        videoUrlInput.focus();
      }
    } catch (err) {
      console.warn('Clipboard access denied or unsupported:', err);
    }
  });

  // Preset pill clicks
  document.querySelectorAll('.preset-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const url = pill.getAttribute('data-url');
      const lang = pill.getAttribute('data-lang');
      if (url) {
        videoUrlInput.value = url;
      }
      if (lang && targetLangSelect) {
        targetLangSelect.value = lang;
        updateLangDirectionBadge();
        updateVoiceOptionsForTarget(lang);
      }
    });
  });

  // Studio Tabs switching
  window.switchStudioTab = function(tabId, clickedBtn) {
    const allTabBtns = document.querySelectorAll('.studio-tab-btn');
    const allTabContents = document.querySelectorAll('.tab-content');
    allTabBtns.forEach((b) => b.classList.remove('active'));
    allTabContents.forEach((c) => c.classList.remove('active'));
    if (clickedBtn) {
      clickedBtn.classList.add('active');
    } else {
      const matchingBtn = document.querySelector(`.studio-tab-btn[data-tab="${tabId}"]`);
      if (matchingBtn) matchingBtn.classList.add('active');
    }
    const target = document.getElementById(tabId);
    if (target) target.classList.add('active');
  };

  if (tabButtons && tabContents) {
    tabButtons.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        const tabId = btn.getAttribute('data-tab');
        window.switchStudioTab(tabId, btn);
      });
    });
  }

  // Toggle Advanced Drawer
  window.toggleStudioConfig = function() {
    if (!configPanel) return;
    const isExpanded = configPanel.classList.toggle('expanded');
    configPanel.classList.toggle('open', isExpanded);
    if (configToggleArrow) {
      configToggleArrow.classList.toggle('rotated', isExpanded);
      configToggleArrow.classList.toggle('open', isExpanded);
    }
  };

  if (toggleConfigBtn && configPanel) {
    toggleConfigBtn.addEventListener('click', (e) => {
      e.preventDefault();
      window.toggleStudioConfig();
    });
  }

  // Live Pacing Sandbox Test
  const btnTestPacing = document.getElementById('btnTestPacing');
  const pacingInputText = document.getElementById('pacingInputText');
  const pacingDurationInput = document.getElementById('pacingDurationInput');
  const pacingResultBox = document.getElementById('pacingResultBox');
  const pacingResultText = document.getElementById('pacingResultText');

  if (btnTestPacing && pacingInputText) {
    btnTestPacing.addEventListener('click', async (e) => {
      e.preventDefault();
      const text = pacingInputText.value.trim();
      const duration = parseFloat(pacingDurationInput?.value || '1.5');
      if (!text) return;
      btnTestPacing.textContent = '⏳ Optimizing...';
      try {
        const res = await fetch('/api/optimize_pacing', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text,
            target_duration: duration,
            target_lang: targetLangSelect?.value || 'en',
          }),
        });
        const data = await res.json();
        if (pacingResultBox && pacingResultText) {
          pacingResultText.textContent = data.optimized_text || text;
          pacingResultBox.classList.remove('hidden');
        }
      } catch (err) {
        console.warn('Pacing sandbox error:', err);
      } finally {
        btnTestPacing.textContent = '⚡ Test AI Optimization';
      }
    });
  }

  // Speaker mode and auto gender toggles
  function updateVoiceGroupVisibility() {
    if (!autoGenderToggle.checked) {
      maleVoiceGroup.classList.add('hidden');
      femaleVoiceGroup.classList.add('hidden');
      singleVoiceGroup.classList.remove('hidden');
      return;
    }
    singleVoiceGroup.classList.add('hidden');
    const mode = speakerModeSelect ? speakerModeSelect.value : 'auto';
    if (mode === 'male_only') {
      maleVoiceGroup.classList.remove('hidden');
      femaleVoiceGroup.classList.add('hidden');
    } else if (mode === 'female_only') {
      maleVoiceGroup.classList.add('hidden');
      femaleVoiceGroup.classList.remove('hidden');
    } else {
      maleVoiceGroup.classList.remove('hidden');
      femaleVoiceGroup.classList.remove('hidden');
    }
  }

  autoGenderToggle.addEventListener('change', updateVoiceGroupVisibility);
  if (speakerModeSelect) {
    speakerModeSelect.addEventListener('change', updateVoiceGroupVisibility);
  }

  // Audio mix slider
  mixOriginalSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    mixValBadge.textContent = val === 0 ? '0% (Clean Dub)' : `${val}%`;
  });

  // Refresh gallery
  refreshGalleryBtn.addEventListener('click', () => {
    loadVideoGallery();
  });

  // Submit Dubbing Form
  dubForm.addEventListener('submit', (e) => {
    e.preventDefault();
    startDubbing();
  });

  // Script Proofreading Modal open/close
  if (openProofreadManualBtn) {
    openProofreadManualBtn.addEventListener('click', () => openProofreadModal());
  }
  if (openProofreadFromPlayerBtn) {
    openProofreadFromPlayerBtn.addEventListener('click', () => openProofreadModal());
  }
  if (closeProofreadBtn) {
    closeProofreadBtn.addEventListener('click', () => closeProofreadModal());
  }
  if (cancelProofreadBtn) {
    cancelProofreadBtn.addEventListener('click', () => closeProofreadModal());
  }
  if (saveProofreadBtn) {
    saveProofreadBtn.addEventListener('click', () => saveProofreadEdits());
  }
  if (redubModalBtn) {
    redubModalBtn.addEventListener('click', () => applyAndRedubVideo());
  }
  if (autoOptimizeAllBtn) {
    autoOptimizeAllBtn.addEventListener('click', () => autoOptimizeAllSegments());
  }
  const setAllMaleBtn = document.getElementById('setAllMaleBtn');
  const setAllFemaleBtn = document.getElementById('setAllFemaleBtn');
  if (setAllMaleBtn) {
    setAllMaleBtn.addEventListener('click', () => {
      if (!currentSegments || !currentSegments.length) return;
      currentSegments.forEach(s => s.gender = 'male');
      renderProofreadEditor(currentSegments);
    });
  }
  if (setAllFemaleBtn) {
    setAllFemaleBtn.addEventListener('click', () => {
      if (!currentSegments || !currentSegments.length) return;
      currentSegments.forEach(s => s.gender = 'female');
      renderProofreadEditor(currentSegments);
    });
  }
  if (proofreadModal) {
    proofreadModal.addEventListener('click', (e) => {
      if (e.target === proofreadModal) {
        closeProofreadModal();
      }
    });
  }
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && proofreadModal && !proofreadModal.classList.contains('hidden')) {
      closeProofreadModal();
    }
  });

  // Synced Transcript Time Update
  if (mainVideoPlayer) {
    mainVideoPlayer.addEventListener('timeupdate', () => {
      const curTime = mainVideoPlayer.currentTime;
      const items = syncedTranscriptList.querySelectorAll('.synced-line-item');
      items.forEach(item => {
        const start = parseFloat(item.dataset.start);
        const end = parseFloat(item.dataset.end);
        if (curTime >= start && curTime <= end) {
          if (!item.classList.contains('active')) {
            items.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }
        }
      });
    });
  }

  // --- Core API Functions ---

  /**
   * Fetch all supported Indian & global languages.
   */
  async function loadLanguages() {
    try {
      const res = await fetch('/api/languages');
      if (!res.ok) throw new Error('Failed to fetch languages');
      const data = await res.json();
      allLanguages = data.languages || [];

      if (targetLangSelect) {
        targetLangSelect.innerHTML = '';

        const indianGroup = document.createElement('optgroup');
        indianGroup.label = '🇮🇳 Indian Languages';

        const globalGroup = document.createElement('optgroup');
        globalGroup.label = '🌍 Global Languages';

        allLanguages.forEach(lang => {
          const opt = document.createElement('option');
          opt.value = lang.code;
          opt.textContent = lang.name;
          opt.dataset.male = lang.male_voice || '';
          opt.dataset.female = lang.female_voice || '';
          opt.dataset.region = lang.region || 'Global';

          if (lang.region === 'Indian') {
            indianGroup.appendChild(opt);
          } else {
            globalGroup.appendChild(opt);
          }
        });

        targetLangSelect.appendChild(indianGroup);
        targetLangSelect.appendChild(globalGroup);
        targetLangSelect.value = 'en';
        updateLangDirectionBadge();
      }
    } catch (err) {
      console.warn('Could not load language catalog:', err);
    }
  }

  /**
   * Automatically select paired native male/female voices when target language changes.
   */
  function updateVoiceOptionsForTarget(langCode) {
    const matched = allLanguages.find(l => l.code === langCode);
    if (!matched) return;

    if (matched.male_voice) {
      let optExists = Array.from(maleVoiceSelect.options).some(o => o.value === matched.male_voice);
      if (!optExists) {
        const opt = new Option(`${matched.male_voice} (${matched.name} Male)`, matched.male_voice, true, true);
        maleVoiceSelect.add(opt);
      }
      maleVoiceSelect.value = matched.male_voice;
    }

    if (matched.female_voice) {
      let optExists = Array.from(femaleVoiceSelect.options).some(o => o.value === matched.female_voice);
      if (!optExists) {
        const opt = new Option(`${matched.female_voice} (${matched.name} Female)`, matched.female_voice, true, true);
        femaleVoiceSelect.add(opt);
      }
      femaleVoiceSelect.value = matched.female_voice;
      singleVoiceSelect.value = matched.female_voice;
    }
  }

  /**
   * Fetch full list of voices from backend.
   */
  async function loadAvailableVoices() {
    try {
      const res = await fetch('/api/voices');
      if (!res.ok) throw new Error('Failed to fetch voices');
      const data = await res.json();
      const voices = data.voices || [];

      maleVoiceSelect.innerHTML = '';
      femaleVoiceSelect.innerHTML = '';
      singleVoiceSelect.innerHTML = '';

      voices.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v.id;
        opt.textContent = `${v.id} (${v.gender} - ${v.description || v.language})`;

        if (v.gender.toLowerCase() === 'male') {
          maleVoiceSelect.appendChild(opt.cloneNode(true));
        } else {
          femaleVoiceSelect.appendChild(opt.cloneNode(true));
        }
        singleVoiceSelect.appendChild(opt.cloneNode(true));
      });

      if (Array.from(maleVoiceSelect.options).some(o => o.value === 'en-US-ChristopherNeural')) {
        maleVoiceSelect.value = 'en-US-ChristopherNeural';
      }
      if (Array.from(femaleVoiceSelect.options).some(o => o.value === 'en-US-JennyNeural')) {
        femaleVoiceSelect.value = 'en-US-JennyNeural';
        singleVoiceSelect.value = 'en-US-JennyNeural';
      }
    } catch (err) {
      console.warn('Could not load voice catalogue:', err);
    }
  }

  /**
   * Fetch gallery list of previously dubbed videos.
   */
  async function loadVideoGallery() {
    try {
      const res = await fetch('/api/videos');
      if (!res.ok) throw new Error('Failed to fetch video gallery');
      const data = await res.json();
      renderGalleryGrid(data.videos || []);
    } catch (err) {
      galleryGrid.innerHTML = `
        <div class="loading-placeholder">
          Failed to load video library. Check server connection.
        </div>`;
    }
  }

  /**
   * Render video cards into the gallery.
   */
  function renderGalleryGrid(videos) {
    if (!videos.length) {
      galleryGrid.innerHTML = `
        <div class="loading-placeholder">
          No dubbed videos found. Enter a video URL above to create your first dub!
        </div>`;
      return;
    }

    galleryGrid.innerHTML = '';
    videos.forEach(v => {
      const card = document.createElement('div');
      card.className = 'glass-card gallery-card';

      const stem = v.filename.replace('.mp4', '');
      const cleanTitle = stem.replace(/_/g, ' ');

      card.innerHTML = `
        <div class="gallery-thumbnail">
          <video src="${v.url}#t=1.0" preload="metadata" muted playsinline></video>
          <div class="play-overlay">
            <svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          </div>
          <span class="gallery-size-tag">${v.size_mb} MB</span>
        </div>
        <div class="gallery-info">
          <h4 class="gallery-title" title="${cleanTitle}">${cleanTitle}</h4>
          <div class="gallery-meta">
            <span class="gallery-date">${v.created_at || 'Recent'}</span>
            <div class="gallery-actions">
              ${v.srt_url ? `<a href="${v.srt_url}" download class="btn-icon-sm" title="Download Subtitles">CC</a>` : ''}
              <a href="${v.url}" download class="btn-icon-sm" title="Download Video">⬇</a>
              <button type="button" class="btn-icon-sm btn-delete-video" title="Delete Dubbed Video" data-filename="${v.filename}" aria-label="Delete video">🗑️</button>
            </div>
          </div>
        </div>
      `;

      card.querySelector('.gallery-thumbnail').addEventListener('click', () => {
        loadVideoIntoPlayer(v.url, v.srt_url, cleanTitle, `${v.size_mb} MB • Ready`, v.vtt_url, v.bilingual_srt_url);
        showcaseSection.classList.remove('hidden');
        showcaseSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });

      const delBtn = card.querySelector('.btn-delete-video');
      if (delBtn) {
        delBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          deleteDubbedVideo(v.filename);
        });
      }

      galleryGrid.appendChild(card);
    });
  }

  /**
   * Delete a dubbed video and its associated subtitles.
   */
  async function deleteDubbedVideo(filename) {
    if (!filename) return;
    const cleanName = filename.split('?')[0].split('/').pop();
    const confirmed = window.confirm(`Are you sure you want to permanently delete "${cleanName}" and its subtitles?`);
    if (!confirmed) return;

    try {
      const res = await fetch(`/api/videos/${encodeURIComponent(cleanName)}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to delete video');
      }

      // If this video is currently active in the player, reset the player
      if (currentVideoUrl && currentVideoUrl.includes(cleanName)) {
        mainVideoPlayer.pause();
        mainVideoPlayer.src = '';
        currentVideoUrl = null;
        currentSrtUrl = null;
        currentVttUrl = null;
        currentBilingualSrtUrl = null;
        playerVideoTitle.textContent = 'No Video Selected';
        playerVideoMeta.innerHTML = '';
        showcaseSection.classList.add('hidden');
      }

      // Refresh video library
      await loadVideoGallery();

      // Show temporary notice
      showDeleteToast(`Deleted ${cleanName}`);
    } catch (err) {
      alert(`Could not delete video: ${err.message}`);
    }
  }

  /**
   * Display floating toast notification when a video is deleted.
   */
  function showDeleteToast(message) {
    let toast = document.getElementById('deleteToast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'deleteToast';
      toast.style.cssText = `
        position: fixed;
        bottom: 28px;
        right: 28px;
        background: rgba(15, 23, 42, 0.95);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(8px);
        padding: 12px 20px;
        border-radius: 10px;
        font-size: 0.9rem;
        font-weight: 500;
        z-index: 9999;
        display: flex;
        align-items: center;
        gap: 8px;
        transition: opacity 0.3s ease, transform 0.3s ease;
      `;
      document.body.appendChild(toast);
    }
    toast.innerHTML = `<span>🗑️</span> <span>${message}</span>`;
    toast.style.opacity = '1';
    toast.style.transform = 'translateY(0)';
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
    }, 3200);
  }

  /**
   * Load video and subtitles into HTML5 player.
   */
  function loadVideoIntoPlayer(videoUrl, srtUrl, title, metaStr, vttUrl, bilingualSrtUrl) {
    currentVideoUrl = videoUrl;
    currentSrtUrl = srtUrl;
    currentVttUrl = vttUrl;
    currentBilingualSrtUrl = bilingualSrtUrl;

    mainVideoPlayer.src = videoUrl;
    playerVideoTitle.textContent = title;

    if (metaStr) {
      playerVideoMeta.innerHTML = `<span class="badge">${metaStr}</span>`;
    } else {
      playerVideoMeta.innerHTML = '';
    }

    downloadVideoBtn.href = videoUrl;
    if (srtUrl) {
      downloadSrtBtn.href = srtUrl;
      downloadSrtBtn.classList.remove('hidden');
      subtitlesTrack.src = srtUrl;
    } else {
      downloadSrtBtn.classList.add('hidden');
    }

    if (vttUrl && downloadVttBtn) {
      downloadVttBtn.href = vttUrl;
      downloadVttBtn.classList.remove('hidden');
    }

    if (bilingualSrtUrl && downloadBilingualBtn) {
      downloadBilingualBtn.href = bilingualSrtUrl;
      downloadBilingualBtn.classList.remove('hidden');
    }

    mainVideoPlayer.load();
    mainVideoPlayer.play().catch(() => {});
  }

  /**
   * Trigger the dubbing pipeline via POST /api/dub and listen via EventSource.
   */
  async function startDubbing() {
    const url = videoUrlInput.value.trim();
    if (!url) {
      videoUrlInput.focus();
      return;
    }

    // Set UI to loading state
    setLoadingState(true);
    resetStepperUI();
    pipelineSection.classList.remove('hidden');
    pipelineErrorBanner.classList.add('hidden');
    pipelineSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    startOverallTimer();

    const payload = {
      url: url,
      source_lang: sourceLangSelect ? sourceLangSelect.value : 'auto',
      target_lang: targetLangSelect ? targetLangSelect.value : 'en',
      model: whisperModelSelect.value,
      auto_gender: autoGenderToggle.checked,
      speaker_mode: speakerModeSelect ? speakerModeSelect.value : 'auto',
      male_voice: maleVoiceSelect.value,
      female_voice: femaleVoiceSelect.value,
      voice: singleVoiceSelect.value,
      mix_original: parseFloat(mixOriginalSlider.value) / 100.0,
      embed_subtitles: embedSubtitlesToggle.checked,
      subtitle_mode: subtitleModeSelect ? subtitleModeSelect.value : 'soft',
      lip_sync_mode: lipSyncModeSelect ? lipSyncModeSelect.value : 'visual_adaptive',
      enable_voice_preservation: document.getElementById('voicePreserveToggle')?.checked ?? true,
      voice_preservation_mode: document.getElementById('voicePreserveModeSelect')?.value || 'adaptive_prosody',
      enable_lipsync: document.getElementById('lipsyncToggle')?.checked ?? true,
      lipsync_model: document.getElementById('lipsyncModelSelect')?.value || 'wav2lip',
    };

    try {
      const res = await fetch('/api/dub', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to initialize dubbing job');
      }

      const data = await res.json();
      const jobId = data.job_id;
      activeJobId.textContent = `Job: #${jobId}`;

      connectEventStream(jobId);
    } catch (err) {
      showPipelineError('Job Initialization Failed', err.message);
      setLoadingState(false);
      stopOverallTimer();
    }
  }

  /**
   * Connect to Server-Sent Events stream for real-time progress.
   */
  function connectEventStream(jobId) {
    if (currentEventSource) {
      currentEventSource.close();
    }

    const streamUrl = `/api/dub/stream/${jobId}`;
    currentEventSource = new EventSource(streamUrl);

    currentEventSource.onmessage = (e) => {
      if (!e.data) return;
      try {
        const evt = JSON.parse(e.data);
        handlePipelineEvent(evt);
      } catch (parseErr) {
        console.warn('Failed to parse SSE payload:', e.data, parseErr);
      }
    };

    currentEventSource.onerror = async (err) => {
      console.warn('EventSource closed or completed by server', err);
      if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
      }
      // Safety fallback check: verify if the job has finished on the server
      try {
        const res = await fetch(`/api/dub/status/${jobId}`);
        if (res.ok) {
          const st = await res.json();
          if (st.status === 'finished' && st.latest_event) {
            handlePipelineEvent(st.latest_event);
          }
        }
      } catch (checkErr) {
        console.warn('Status fallback check error:', checkErr);
      }
    };
  }

  /**
   * Process individual stage events from the backend pipeline.
   */
  function handlePipelineEvent(evt) {
    const stage = evt.stage;
    const status = evt.status;

    if (stage === -1 || status === 'error') {
      showPipelineError(evt.error || 'Dubbing Pipeline Error', evt.traceback || '');
      setLoadingState(false);
      stopOverallTimer();
      if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
      }
      return;
    }

    // Stage 1: Download
    if (stage === 1) {
      if (status === 'running') {
        updateStepCard(1, 'running', 10, '10%', 'Downloading video stream...');
      } else if (status === 'downloading') {
        const pct = evt.percent || 0;
        updateStepCard(1, 'running', pct, `${pct}%`, 'Extracting audio track...');
      } else if (status === 'done') {
        updateStepCard(1, 'done', 100, '100%', `Completed (${evt.elapsed || 0}s)`);
      }
    }

    // Stage 2: Transcribe & Translate
    else if (stage === 2) {
      if (status === 'running') {
        updateStepCard(2, 'running', 15, '15%', 'Whisper AI transcribing...');
      } else if (status === 'translating') {
        const pct = evt.percent || 50;
        updateStepCard(2, 'running', pct, `${pct}%`, 'Translating segments...');
      } else if (status === 'done') {
        updateStepCard(2, 'done', 100, '100%', `Extracted ${evt.segments?.length || 0} segments`);
        if (evt.segments && evt.segments.length > 0) {
          currentSegments = evt.segments;
          renderSegmentsTable(evt.segments);
          renderSyncedTranscript(evt.segments);
        }
      }
    }

    // Stage 2.8: Speaker Voice Preservation
    else if (stage === 2.8) {
      updateStepCard(3, 'running', 20, '20%', 'Profiling speaker pitch (F0) & tone...');
    }

    // Stage 3: Synthesizing Speech
    else if (stage === 3) {
      if (status === 'running') {
        updateStepCard(3, 'running', 25, '25%', 'Voice-Preserved Neural Speech...');
      } else if (status === 'synthesizing') {
        const pct = evt.percent || 0;
        updateStepCard(3, 'running', pct, `${pct}%`, `Synthesizing ${evt.completed || 0} / ${evt.total || 0}`);
      } else if (status === 'done') {
        updateStepCard(3, 'done', 100, '100%', `Voice Preserved (${evt.elapsed || 0}s)`);
      }
    }

    // Stage 4: Lip-Sync Alignment
    else if (stage === 4) {
      if (status === 'running') {
        updateStepCard(4, 'running', 15, '15%', 'Stripping silence & tempo stretching...');
      } else if (status === 'syncing') {
        const pct = evt.percent || 0;
        updateStepCard(4, 'running', pct, `${pct}%`, `Aligned ${evt.completed || 0} / ${evt.total || 0}`);
      } else if (status === 'done') {
        updateStepCard(4, 'running', 60, '60%', `Audio Aligned (${evt.elapsed || 0}s)`);
      }
    }

    // Stage 4.5: AI Lip Synchronization (Wav2Lip)
    else if (stage === 4.5) {
      if (status === 'running') {
        updateStepCard(4, 'running', 40, '40%', 'Wav2Lip AI analyzing speech frames...');
      } else if (status === 'lip_syncing') {
        const pct = evt.percent || 50;
        updateStepCard(4, 'running', pct, `${pct}%`, `Lip-Sync: ${evt.synced_frames || 0} frames`);
      }
    }

    // Stage 5: Video Remux
    else if (stage === 5) {
      if (status === 'running') {
        updateStepCard(5, 'running', 50, '50%', 'FFmpeg losslessly remuxing MP4...');
      } else if (status === 'done' || status === 'complete') {
        const remuxSec = evt.elapsed || evt.summary?.remux_time || '1.5';
        updateStepCard(5, 'done', 100, '100%', `Dubbed Video Ready (${remuxSec}s)`);
        setLoadingState(false);
        stopOverallTimer();

        if (currentEventSource) {
          currentEventSource.close();
          currentEventSource = null;
        }

        if (evt.summary) {
          displaySummaryMetrics(evt.summary);
        }

        if (evt.video_url) {
          const videoTitle = evt.summary?.title || 'Dubbed Video';
          const metaStr = `${(evt.summary?.language || 'AUTO').toUpperCase()} → ${(evt.summary?.target_lang || 'EN').toUpperCase()} • ${evt.summary?.file_size_mb || 0} MB`;
          loadVideoIntoPlayer(evt.video_url, evt.srt_url, videoTitle, metaStr, evt.vtt_url, evt.bilingual_srt_url);
        }

        showcaseSection.classList.remove('hidden');
        showcaseSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        loadVideoGallery();
      }
    }
  }

  function updateStepCard(stepNumber, state, percent, pctText, statText) {
    const card = document.getElementById(`stepCard${stepNumber}`);
    if (!card) return;
    const bar = document.getElementById(`progressBar${stepNumber}`);
    const pct = document.getElementById(`stepPct${stepNumber}`);
    const stat = document.getElementById(`stepStat${stepNumber}`);
    const badge = card.querySelector('.step-status-badge');

    if (state === 'running') {
      card.classList.add('active');
      card.classList.remove('completed');
      if (badge) {
        badge.className = 'step-status-badge badge-running';
        badge.textContent = 'Running';
      }
    } else if (state === 'done') {
      card.classList.remove('active');
      card.classList.add('completed');
      if (badge) {
        badge.className = 'step-status-badge badge-done';
        badge.textContent = 'Done';
      }
    }

    if (bar) bar.style.width = `${percent}%`;
    if (pct) pct.textContent = pctText;
    if (stat) stat.textContent = statText;
  }

  /**
   * Render segments into the table.
   */
  function renderSegmentsTable(segments) {
    segmentsSection.classList.remove('hidden');
    segmentCountBadge.textContent = `${segments.length} Segments`;
    segmentsTableBody.innerHTML = '';

    segments.forEach((seg, idx) => {
      const row = document.createElement('tr');
      const isMale = (seg.gender || '').toLowerCase() === 'male';

      const genderBadge = isMale 
        ? `<span class="badge badge-male">👦 Boy (Male)</span>` 
        : `<span class="badge badge-female">👧 Girl (Female)</span>`;

      const isPersonLocked = seg.detection_source && seg.detection_source.includes('person_locked');
      const sourceBadge = isPersonLocked
        ? `<span class="badge badge-accent">🎯 Person Locked</span>`
        : (seg.face_detected || seg.visual_applied)
          ? `<span class="badge badge-visual">👁️ Visual Face</span>`
          : `<span class="badge badge-audio">🔊 Vocal Pitch</span>`;

      row.innerHTML = `
        <td>${seg.id || idx + 1}</td>
        <td><code>${formatTimestamp(seg.start)} ➔ ${formatTimestamp(seg.end)}</code></td>
        <td>${genderBadge}</td>
        <td>${sourceBadge}</td>
        <td><code>${seg.visual_conf ? (seg.visual_conf * 100).toFixed(1) + '%' : (seg.f0 ? Math.round(seg.f0) + ' Hz' : '--')}</code></td>
        <td class="orig-cell">${seg.orig_text || seg.text_en || '--'}</td>
        <td class="dubbed-cell"><strong>${seg.text || '--'}</strong></td>
      `;

      segmentsTableBody.appendChild(row);
    });
  }

  /**
   * Render segments into the interactive Synced Subtitle Transcript bar.
   */
  function renderSyncedTranscript(segments) {
    if (!syncedTranscriptList) return;
    syncedTranscriptList.innerHTML = '';

    segments.forEach(seg => {
      const isMale = (seg.gender || '').toLowerCase() === 'male';
      const item = document.createElement('div');
      item.className = 'synced-line-item';
      item.dataset.start = seg.start;
      item.dataset.end = seg.end;

      item.innerHTML = `
        <span class="synced-time">${formatTimestamp(seg.start)}</span>
        <span class="synced-speaker ${isMale ? 'badge-male' : 'badge-female'}">
          ${isMale ? '👦 Boy' : '👧 Girl'}
        </span>
        <div class="synced-texts">
          ${seg.orig_text && seg.orig_text !== seg.text ? `<div class="synced-orig">${seg.orig_text}</div>` : ''}
          <div class="synced-dub">${seg.text}</div>
        </div>
      `;

      item.addEventListener('click', () => {
        if (mainVideoPlayer) {
          mainVideoPlayer.currentTime = seg.start;
          mainVideoPlayer.play().catch(() => {});
        }
      });

      syncedTranscriptList.appendChild(item);
    });
  }

  /**
   * Interactive Script Proofreader Modal Logic
   */
  function openProofreadModal() {
    if (!currentSegments || !currentSegments.length) {
      currentSegments = [
        {
          id: 1,
          start: 0.5,
          end: 3.2,
          gender: 'male',
          orig_text: "Hey! Are we still meeting up for the project review today?",
          text: "Hey! Are we still meeting up for the project review today?",
          visual_applied: true,
          visual_conf: 0.95
        },
        {
          id: 2,
          start: 3.5,
          end: 6.8,
          gender: 'female',
          orig_text: "Yes, absolutely. I have all the slides and metrics completely prepared.",
          text: "Yes, absolutely. I have all the slides and metrics completely prepared.",
          visual_applied: true,
          visual_conf: 0.98
        }
      ];
    }
    proofreadCountBadge.textContent = `${currentSegments.length} Segments`;
    renderProofreadEditor(currentSegments);
    proofreadModal.classList.remove('hidden');
  }

  function closeProofreadModal() {
    proofreadModal.classList.add('hidden');
  }

  function renderProofreadEditor(segments) {
    proofreadList.innerHTML = '';
    segments.forEach((seg, idx) => {
      const isMale = (seg.gender || '').toLowerCase() === 'male';
      const duration = (seg.end - seg.start).toFixed(2);
      const card = document.createElement('div');
      card.className = 'proofread-segment-card';
      card.dataset.index = idx;

      card.innerHTML = `
        <div class="segment-meta-row">
          <span class="segment-time-badge">#${seg.id || idx + 1} &bull; ${formatTimestamp(seg.start)} ➔ ${formatTimestamp(seg.end)} (${duration}s)</span>
          <button type="button" class="speaker-voice-btn" data-index="${idx}">
            ${isMale ? '👦 Boy (Male Voice)' : '👧 Girl (Female Voice)'} &bull; Click to Swap
          </button>
        </div>
        <div class="proofread-texts-grid">
          <div class="orig-text-preview">
            <strong>Original Speech:</strong><br>
            ${seg.orig_text || seg.text_en || '(No original transcript)'}
          </div>
          <div class="dub-text-editor">
            <label style="font-size:0.75rem; font-weight:600; color:#a5b4fc;">Dubbed Target Translation:</label>
            <textarea class="dub-textarea" rows="2" data-index="${idx}">${seg.text || ''}</textarea>
            <div class="dub-editor-footer">
              <button type="button" class="optimize-single-btn" data-index="${idx}">
                ✨ Optimize Pacing (${duration}s slot)
              </button>
            </div>
          </div>
        </div>
      `;

      // Speaker Voice Swap
      card.querySelector('.speaker-voice-btn').addEventListener('click', (e) => {
        const i = parseInt(e.currentTarget.dataset.index, 10);
        const currGender = currentSegments[i].gender;
        currentSegments[i].gender = currGender === 'male' ? 'female' : 'male';
        renderProofreadEditor(currentSegments);
      });

      // Individual Pacing Optimization
      card.querySelector('.optimize-single-btn').addEventListener('click', async (e) => {
        const i = parseInt(e.currentTarget.dataset.index, 10);
        const textarea = card.querySelector('.dub-textarea');
        const textToOpt = textarea.value.trim();
        const slotDur = currentSegments[i].end - currentSegments[i].start;

        try {
          const res = await fetch('/api/optimize_pacing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: textToOpt,
              target_duration: slotDur,
              target_lang: targetLangSelect ? targetLangSelect.value : 'en',
            }),
          });
          if (res.ok) {
            const data = await res.json();
            textarea.value = data.optimized_text;
            currentSegments[i].text = data.optimized_text;
          }
        } catch (err) {
          console.warn('Pacing optimization error:', err);
        }
      });

      // Update segment text on edit
      card.querySelector('.dub-textarea').addEventListener('input', (e) => {
        const i = parseInt(e.currentTarget.dataset.index, 10);
        currentSegments[i].text = e.currentTarget.value;
      });

      proofreadList.appendChild(card);
    });
  }

  async function autoOptimizeAllSegments() {
    for (let i = 0; i < currentSegments.length; i++) {
      const seg = currentSegments[i];
      const slotDur = seg.end - seg.start;
      try {
        const res = await fetch('/api/optimize_pacing', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: seg.text,
            target_duration: slotDur,
            target_lang: targetLangSelect ? targetLangSelect.value : 'en',
          }),
        });
        if (res.ok) {
          const data = await res.json();
          currentSegments[i].text = data.optimized_text;
        }
      } catch (err) {}
    }
    renderProofreadEditor(currentSegments);
  }

  function saveProofreadEdits() {
    renderSegmentsTable(currentSegments);
    renderSyncedTranscript(currentSegments);
    closeProofreadModal();
  }

  async function applyAndRedubVideo() {
    if (!currentSegments || !currentSegments.length) {
      alert('No speech segments available to re-dub.');
      return;
    }

    // Determine current video filename
    let videoFilename = '';
    if (videoPlayer && videoPlayer.src) {
      try {
        const srcUrl = new URL(videoPlayer.src, window.location.origin);
        videoFilename = srcUrl.pathname.split('/').pop();
      } catch (e) {
        videoFilename = '';
      }
    }
    if (!videoFilename || videoFilename === 'undefined') {
      const urlVal = (urlInput ? urlInput.value : '').trim();
      if (urlVal.includes('GdUMxKyqrSs')) {
        videoFilename = 'GdUMxKyqrSs_dubbed_en.mp4';
      } else {
        videoFilename = 'dubbed_video.mp4';
      }
    }

    // Update button loading state
    if (redubModalBtn) redubModalBtn.disabled = true;
    if (redubBtnText) redubBtnText.textContent = '⚡ Re-Dubbing Audio...';
    if (redubBtnSpinner) redubBtnSpinner.classList.remove('hidden');

    try {
      const res = await fetch('/api/redub', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_filename: videoFilename,
          segments: currentSegments,
          mix_original: mixOriginalRange ? parseFloat(mixOriginalRange.value) : 0.0,
          target_lang: targetLangSelect ? targetLangSelect.value : 'en',
          burn_subtitles: subtitleModeSelect ? subtitleModeSelect.value === 'burn' : false,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Re-dubbing failed.');
      }

      const data = await res.json();
      console.log('Re-dub complete:', data);

      // Refresh video player with cache-busting timestamp
      const cacheBust = `?t=${Date.now()}`;
      if (videoPlayer) {
        videoPlayer.src = `${data.video_url}${cacheBust}`;
        videoPlayer.load();
      }

      // Update subtitle track
      if (subTrack && data.vtt_url) {
        subTrack.src = `${data.vtt_url}${cacheBust}`;
      }

      // Update transcript displays
      renderSegmentsTable(currentSegments);
      renderSyncedTranscript(currentSegments);

      // Update metrics
      const maleSegs = currentSegments.filter(s => s.gender === 'male').length;
      const femaleSegs = currentSegments.filter(s => s.gender === 'female').length;
      if (metricSpeakers) {
        metricSpeakers.textContent = `${maleSegs} Male ♂  •  ${femaleSegs} Female ♀`;
      }

      // Refresh video library
      loadVideoLibrary();

      closeProofreadModal();

      if (pipelineErrorBanner) {
        pipelineErrorBanner.classList.add('hidden');
      }
      alert(`🎉 Re-dubbed video generated successfully in ${data.elapsed || 'a few'}s!\nAudio voices and duration are updated.`);

    } catch (err) {
      console.error('Re-dub error:', err);
      alert(`Re-dubbing error: ${err.message}`);
    } finally {
      if (redubModalBtn) redubModalBtn.disabled = false;
      if (redubBtnText) redubBtnText.textContent = '⚡ Apply & Re-Dub Video';
      if (redubBtnSpinner) redubBtnSpinner.classList.add('hidden');
    }
  }

  function displaySummaryMetrics(summary) {
    if (metricSourceLang) metricSourceLang.textContent = (summary.source_language || summary.language || 'Te').toUpperCase();
    if (metricLang) metricLang.textContent = (summary.target_lang || 'En').toUpperCase();
    if (metricSpeakers) {
      metricSpeakers.textContent = `${summary.male_segments || 0} Male ♂  •  ${summary.female_segments || 0} Female ♀`;
    }
    if (metricSegments) metricSegments.textContent = `${summary.speech_segments || 0} segments`;
    if (metricTotalTime) metricTotalTime.textContent = `${summary.total_time || 0}s`;
    const metricVoicePreserve = document.getElementById('metricVoicePreserve');
    const metricLipSync = document.getElementById('metricLipSync');
    if (metricVoicePreserve) {
      const mode = summary.voice_preservation || 'adaptive_prosody';
      metricVoicePreserve.textContent = mode !== 'none' ? 'F0 Pitch & Tone Active' : 'Off';
    }
    if (metricLipSync) {
      if (summary.lipsync_enabled) {
        const stats = summary.lipsync_stats;
        metricLipSync.textContent = stats ? `Wav2Lip (${stats.synced_frames || 0} frames)` : 'Wav2Lip Synchronized';
      } else {
        metricLipSync.textContent = 'Audio Timed';
      }
    }
    if (metricEfficiency) {
      const rtf = summary.rtf ? `${summary.rtf}x` : '< 1.0x';
      metricEfficiency.textContent = `${rtf} Real-Time Factor`;
    }
  }

  function showPipelineError(title, details) {
    errorTitle.textContent = title;
    errorDetails.textContent = details;
    pipelineErrorBanner.classList.remove('hidden');
    pipelineErrorBanner.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function resetStepperUI() {
    for (let i = 1; i <= 5; i++) {
      const card = document.getElementById(`stepCard${i}`);
      if (!card) continue;
      card.className = 'step-card';
      const badge = card.querySelector('.step-status-badge');
      if (badge) {
        badge.className = 'step-status-badge badge-pending';
        badge.textContent = 'Pending';
      }
      const bar = document.getElementById(`progressBar${i}`);
      if (bar) bar.style.width = '0%';
      const pct = document.getElementById(`stepPct${i}`);
      if (pct) pct.textContent = '0%';
      const stat = document.getElementById(`stepStat${i}`);
      if (stat) stat.textContent = 'Waiting...';
    }
  }

  function setLoadingState(isLoading) {
    if (isLoading) {
      startDubBtn.disabled = true;
      btnText.classList.add('hidden');
      btnSpinner.classList.remove('hidden');
    } else {
      startDubBtn.disabled = false;
      btnText.classList.remove('hidden');
      btnSpinner.classList.add('hidden');
    }
  }

  function startOverallTimer() {
    stopOverallTimer();
    timerSeconds = 0;
    overallTimer.textContent = '00:00';
    timerInterval = setInterval(() => {
      timerSeconds++;
      const mins = Math.floor(timerSeconds / 60).toString().padStart(2, '0');
      const secs = (timerSeconds % 60).toString().padStart(2, '0');
      overallTimer.textContent = `${mins}:${secs}`;
    }, 1000);
  }

  function stopOverallTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
  }

  function formatTimestamp(seconds) {
    if (seconds == null || isNaN(seconds)) return '00:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
  }
});
