<template>
    <section class="framework">
        <h3>Framework</h3>
        <div>
            <img v-for="image in images" :src="image">
            <br>
            <p>{{ description }}</p>
        </div>

    </section>
</template>

<script setup lang="ts">
interface Props {
    images?: string[]
    description?: string
}

const { props } = defineProps<{ props: Props }>()
const images = (props.images || []).map(image => (image || "").startsWith("assets") ? new URL(`../${image}`, import.meta.url).href : image)
const description = props.description
</script>

<style lang="scss" scoped>
.framework {
    * {
        max-width: 960px;
        @apply w-full;
    }

}

img {
    @apply my-2;
}
</style>
