import { attachVideo, detachVideo, inVisible } from '@/utils/video'

const videos = new Map<HTMLVideoElement, DOMRect>()

function playOrPause(video: HTMLVideoElement) {
  if (inVisible(video)) {
    // video.play()
  } else {
    video.pause()
  }
}

const onscroll = (evt: Event) => {
  for (const video of videos.keys()) {
    if (video.getAttribute('data-src')) {
      if (inVisible(video)) {
        attachVideo(video)
      }
    }
    playOrPause(video)
  }
}

export default {
  name: 'lazy',
  option: {
    mounted: (el: HTMLElement) => {
      if (el instanceof HTMLVideoElement) {
        const rect = el.getBoundingClientRect()
        videos.set(el, rect)
        // 如果视频在页面窗口内，则加载，否则detach
        if (inVisible(el)) {
          attachVideo(el)
        } else {
          detachVideo(el)
        }
        el.oncanplay = () => {
          videos.set(el, el.getBoundingClientRect())
          playOrPause(el)
        }
      }
      if (videos.size) {
        !window.onscroll && (window.onscroll = onscroll)
      }
    },
    unmounted: (el: HTMLElement) => {
      if (el instanceof HTMLVideoElement) {
        videos.delete(el)
      }
      if (!videos.size) {
        window.onscroll = null
      }
    }
  }
}
