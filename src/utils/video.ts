export function inVisible(video: HTMLElement) {
  const { left, right, top, bottom, width, height } = video.getBoundingClientRect()
  if (bottom < 0 || top > window.innerHeight) {
    return false
  } else if (left != 0 && right != 0) {
    return true
  }
  return false
}


export function detachVideo(video: HTMLVideoElement) {
  const src = video.src
  video.src = ''
  video.setAttribute('data-src', src)
}

export function attachVideo(video: HTMLVideoElement) {
  if (!video.getAttribute('data-src')) return
  const src = video.getAttribute('data-src')!
  video.src = src
  video.removeAttribute('data-src')
}

