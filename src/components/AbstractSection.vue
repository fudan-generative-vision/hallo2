<template>
    <section class="abstract">
        <div>
            <h3>Abstract</h3>
            <div v-if="figure" class="figure">
                <img :src="figure">
            </div>
            <div v-if="video" class="figure">
                <video :src="video" />
            </div>
            <p>{{ props.content }}</p>
        </div>
    </section>
</template>

<script setup lang="ts">
interface Props {
    content?: string
    figure?: string
    video?: string
}
const { props } = defineProps<{ props: Props }>()
const figure = (props.figure || "").startsWith("assets") ? new URL(`../${props.figure}`, import.meta.url).href : props.figure
const video = (props.video || "").startsWith("assets") ? new URL(`../${props.video}`, import.meta.url).href : props.video

</script>

<style lang="scss" scoped>
.abstract {
    div {
        max-width: 960px;
        @apply w-full mt-2;
    }

    li {
        @apply flex flex-row my-1;

        :first-child {
            @apply mr-2;
        }
    }
}

.figure {
    margin: 1rem 0;
}

p {
    white-space: pre-line;
}
</style>